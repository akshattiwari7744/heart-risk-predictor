"""FastAPI application. Start with:  python -m uvicorn api.main:app --host 127.0.0.1 --port 8000"""
from __future__ import annotations

import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from api import text
from api.model_service import ModelService
from api.schemas import (FIELD_HELP, ErrorResponse, HealthResponse, ModelBrief, ModelInfoResponse, PredictionRequest,
                         PredictionResponse)
from api.settings import ConfigError, Settings

log = logging.getLogger("heart_api")
_RID_RE = re.compile(r"^[A-Za-z0-9-]{8,64}$")
_DOC_PATHS = ("/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect")
_HTTP_MESSAGES = {404: ("not_found", "The requested resource does not exist."),
                  405: ("method_not_allowed", "This HTTP method is not allowed for this resource."),
                  413: ("payload_too_large", "The request body is too large.")}


def _configure_logging(level: str) -> None:
    log.setLevel(level)
    if not log.handlers and not logging.getLogger().handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        log.addHandler(h)


def _rid(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


def _error(status: int, code: str, message: str, rid: str, errors=None) -> JSONResponse:
    body = ErrorResponse(error=code, message=message, request_id=rid, errors=errors).model_dump(exclude_none=True)
    return JSONResponse(status_code=status, content=body)


def _allowed_text(field: str) -> Optional[str]:
    h = FIELD_HELP.get(field)
    if not h:
        return None
    if field == "bmi":
        return f"number between {h['min']} and {h['max']}"
    if h["kind"] == "binary":
        return "0 or 1 (whole number)"
    return f"whole number from {int(h['min'])} to {int(h['max'])}"


def _friendly(err: dict) -> dict:
    """Sanitise a pydantic error: field + plain message only. Never echo the submitted value, URLs or internals."""
    loc = [str(x) for x in err.get("loc", ()) if x != "body"]
    name, t = ".".join(loc) or "body", err.get("type", "")
    if t == "json_invalid":
        name = "body"                       # loc would otherwise be the character position
    if name == "body" and t == "missing":
        problem = "request body is required"
    elif t == "json_invalid":
        problem = "request body is not valid JSON"
    elif t in ("model_attributes_type", "dict_type", "model_type"):
        problem = "request body must be a JSON object of the 19 model inputs"
    elif t == "missing":
        problem = "required field is missing"
    elif t == "extra_forbidden":
        problem = "unknown field - not one of the model's 19 inputs"
    elif t.startswith("int_"):
        problem = "must be a whole number (text, true/false, decimals and null are not accepted)"
    elif t.startswith("float_"):
        problem = "must be a number (text, true/false and null are not accepted)"
    elif t in ("greater_than_equal", "less_than_equal"):
        problem = "value is outside the allowed range"
    elif t == "finite_number":
        problem = "must be a finite number"
    else:
        problem = "invalid value"
    out = {"field": name, "problem": problem}
    a = _allowed_text(name)
    if a:
        out["allowed"] = a
    return out


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or Settings.from_env()
    _configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = ModelService.load(settings.artifacts_dir, settings.allow_quick_model)   # once, at startup
        yield

    docs = settings.enable_docs
    app = FastAPI(title="Heart Risk Predictor API", version="0.4.0", description=text.API_DESCRIPTION, lifespan=lifespan,
                  docs_url="/docs" if docs else None, redoc_url="/redoc" if docs else None,
                  openapi_url="/openapi.json" if docs else None)
    app.state.settings = settings
    app.state.service = ModelService(artifacts_dir=settings.artifacts_dir, problem_code="not_started")

    # ---- request guard: request id, size limit, safe 500s, security headers, logging (no health inputs are logged)
    @app.middleware("http")
    async def guard(request: Request, call_next):
        incoming = request.headers.get("x-request-id", "")
        rid = incoming if _RID_RE.match(incoming) else uuid.uuid4().hex
        request.state.request_id = rid
        t0 = time.perf_counter()
        try:
            cl = request.headers.get("content-length")
            if cl is not None and (not cl.isdigit()):
                response = _error(400, "bad_request", "The request could not be understood.", rid)
            elif cl is not None and int(cl) > settings.max_body_bytes:
                response = _error(413, *_HTTP_MESSAGES[413], rid)
            else:
                response = await call_next(request)
        except Exception:
            log.exception("rid=%s unhandled_error %s %s", rid, request.method, request.url.path)     # traceback stays server-side
            response = _error(500, "internal_error", "An internal error occurred. Please try again later.", rid)
        response.headers["X-Request-ID"] = rid
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        if request.url.path not in _DOC_PATHS:
            response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
        if request.url.path == "/predict":
            response.headers["Cache-Control"] = "no-store"
        log.info("rid=%s %s %s -> %s in %.1f ms", rid, request.method, request.url.path, response.status_code,
                 1000 * (time.perf_counter() - t0))
        return response

    if settings.cors_origins:                                  # added last => outermost, so errors also carry CORS headers
        app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins), allow_credentials=False,
                           allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["Content-Type", "X-Request-ID"],
                           expose_headers=["X-Request-ID"], max_age=600)

    # ---- error handlers (uniform shape; no stack traces, paths or submitted values)
    @app.exception_handler(RequestValidationError)
    async def on_validation(request: Request, exc: RequestValidationError):
        items = [_friendly(e) for e in exc.errors()]
        log.info("rid=%s validation_error fields=%s", _rid(request), sorted({i["field"] for i in items}))
        return _error(422, "validation_error", "The request was not accepted. Fix the listed fields and try again.", _rid(request), items)

    @app.exception_handler(StarletteHTTPException)
    async def on_http(request: Request, exc: StarletteHTTPException):
        code, msg = _HTTP_MESSAGES.get(exc.status_code, ("http_error", "The request could not be completed."))
        return _error(exc.status_code, code, msg, _rid(request))

    @app.exception_handler(Exception)
    async def on_unhandled(request: Request, exc: Exception):
        log.error("rid=%s unhandled_error", _rid(request), exc_info=exc)
        return _error(500, "internal_error", "An internal error occurred. Please try again later.", _rid(request))

    # ---- routes
    unavailable = {503: {"model": ErrorResponse, "description": "Model not loaded"}}

    @app.get("/", include_in_schema=False)
    def root():
        return {"service": "Heart Risk Predictor API", "docs": "/docs" if docs else None, "health": "/health", "disclaimer": text.DISCLAIMER}

    @app.get("/health", response_model=HealthResponse, tags=["status"], summary="Is the API running and is the model loaded?")
    def health(request: Request):
        s: ModelService = request.app.state.service
        return HealthResponse(status="ok" if s.loaded else "degraded", model_loaded=s.loaded,
                              model_name=s.model.model_name if s.loaded else None,
                              training_mode=s.training_mode if s.loaded else None, problem_code=None if s.loaded else s.problem_code)

    @app.get("/model-info", response_model=ModelInfoResponse, responses=unavailable, tags=["status"],
             summary="Frozen model details, inputs and intended use")
    def model_info(request: Request):
        s: ModelService = request.app.state.service
        if not s.loaded:
            return _error(503, "model_unavailable", "The prediction model is not available.", _rid(request))
        d = s.decisions
        return ModelInfoResponse(model_name=s.model.model_name, calibration=s.model.calibrator.method, training_mode=s.training_mode,
                                 decision_threshold=s.model.threshold, threshold_rule=str(d.get("threshold_rule", "")),
                                 target_recall_on_validation=(d.get("cfg") or {}).get("target_recall"),
                                 features=s.features_info(), target=text.TARGET_STATEMENT, disclaimer=text.DISCLAIMER, request_id=_rid(request))

    @app.post("/predict", response_model=PredictionResponse, tags=["prediction"],
              responses={422: {"model": ErrorResponse, "description": "Missing or invalid inputs"}, **unavailable,
                         500: {"model": ErrorResponse, "description": "Internal error (details are only in server logs)"}},
              summary="Estimated likelihood of REPORTING existing heart disease (not a diagnosis, not future risk)")
    def predict(payload: PredictionRequest, request: Request):
        s: ModelService = request.app.state.service
        if not s.loaded:
            return _error(503, "model_unavailable", "The prediction model is not available.", _rid(request))
        p, above = s.predict(payload.model_dump())
        log.info("rid=%s predict ok above_threshold=%s", _rid(request), above)                    # no input values logged
        return PredictionResponse(
            estimated_probability=round(p, 6), decision_threshold=s.model.threshold, above_threshold=above,
            result_label="above_decision_threshold" if above else "below_decision_threshold",
            interpretation=text.INTERPRETATION_ABOVE if above else text.INTERPRETATION_BELOW, target=text.TARGET_STATEMENT,
            model=ModelBrief(model_name=s.model.model_name, calibration=s.model.calibrator.method, training_mode=s.training_mode),
            disclaimer=text.DISCLAIMER, request_id=_rid(request))

    return app


app = create_app()
