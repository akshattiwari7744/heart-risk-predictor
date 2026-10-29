"""Phase 4 API tests. Use temp artifacts built from tiny synthetic data; the only production-artifact test is read-only."""
import copy
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.model_service as ms
import config as C
import train_model as T
from api import text
from api.main import create_app
from api.model_service import ModelService
from api.schemas import _EXAMPLE, FEATURE_ORDER, FIELD_HELP
from api.settings import ConfigError, Settings
from test_phase3 import make_frame

TINY = T.Cfg(cv_folds=2, hgb_iter=2, tune_rows=1500, n_boot=15, select_boot=15)


# ----------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def art(tmp_path_factory):
    root = tmp_path_factory.mktemp("api_art")
    data = root / "data"
    data.mkdir()
    for name, (n, seed) in {"train": (3000, 1), "val": (1500, 2), "test": (1500, 3)}.items():
        make_frame(n, seed).to_parquet(data / f"{name}.parquet")
    T.main(["--data-dir", str(data), "--artifacts-dir", str(root / "artifacts"), "--docs-dir", str(root / "docs")], cfg=TINY)
    return root / "artifacts"


@pytest.fixture()
def client(art):
    app = create_app(Settings(artifacts_dir=art))
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def variant(art, tmp_path, edit_meta=None, edit_dec=None, strip_mode=False):
    d = tmp_path / "artifacts"
    shutil.copytree(art, d)
    meta, dec = json.loads((d / "model_meta.json").read_text()), json.loads((d / "decisions_frozen.json").read_text())
    if strip_mode:
        for o in (meta, dec, dec["cfg"]):
            o.pop("training_mode", None)
            o.pop("mode", None)
    if edit_meta:
        edit_meta(meta)
    if edit_dec:
        edit_dec(dec)
    (d / "model_meta.json").write_text(json.dumps(meta))
    (d / "decisions_frozen.json").write_text(json.dumps(dec))
    return d


class StubModel:
    def __init__(self, real, p=None, boom=None):
        self.feature_names, self.threshold, self.model_name, self.calibrator = real.feature_names, real.threshold, real.model_name, real.calibrator
        self._p, self._boom = p, boom

    def predict_proba_pos(self, X):
        if self._boom:
            raise self._boom
        return np.array([self._p])


# ----------------------------------------------------------------------------- model loading / contract
def test_model_loads_and_matches_frozen_artifacts(art):
    svc = ModelService.load(art)
    assert svc.loaded and svc.problem_code is None and svc.training_mode == "full"
    meta, dec = json.loads((art / "model_meta.json").read_text()), json.loads((art / "decisions_frozen.json").read_text())
    assert svc.model.threshold == dec["threshold"] == meta["threshold"]
    assert svc.model.calibrator.method == dec["calibration_method"]


def test_exactly_19_features_in_config_order(art):
    svc = ModelService.load(art)
    cfg = [f["out"] for f in C.FEATURES]
    meta = json.loads((art / "model_meta.json").read_text())
    assert len(cfg) == 19 and FEATURE_ORDER == cfg == list(svc.model.feature_names) == meta["features"]


def test_schema_ranges_match_config_code_maps():
    for f in C.FEATURES:
        h = FIELD_HELP[f["out"]]
        if f["rule"] == "map":
            vals = {v for v in f["codes"].values() if not (isinstance(v, float) and np.isnan(v))}
            assert (min(vals), max(vals)) == (h["min"], h["max"]), f["out"]
            assert vals == set(range(int(h["min"]), int(h["max"]) + 1)), f["out"]
    assert (FIELD_HELP["phys_unhealthy_days"]["min"], FIELD_HELP["phys_unhealthy_days"]["max"]) == (0, 30)
    assert (FIELD_HELP["bmi"]["min"], FIELD_HELP["bmi"]["max"]) == (12.0, 99.99)


def test_model_is_loaded_once_not_per_request(art, monkeypatch):
    calls, orig = [], joblib.load
    monkeypatch.setattr(ms.joblib, "load", lambda *a, **k: (calls.append(a[0]), orig(*a, **k))[1])
    with TestClient(create_app(Settings(artifacts_dir=art)), raise_server_exceptions=False) as c:
        for _ in range(3):
            assert c.post("/predict", json=_EXAMPLE).status_code == 200
    assert len(calls) == 1


@pytest.mark.parametrize("name,kwargs,code", [
    ("tampered hash", dict(edit_meta=lambda m: m.update(model_sha256="0" * 64)), "integrity_check_failed"),
    ("sklearn mismatch", dict(edit_meta=lambda m: m["versions"].update({"scikit-learn": "0.0.1"})), "version_mismatch"),
    ("features reordered", dict(edit_meta=lambda m: m.update(features=list(reversed(m["features"])))), "feature_mismatch"),
    ("threshold mismatch", dict(edit_dec=lambda d: d.update(threshold=d["threshold"] + 0.01)), "threshold_mismatch"),
    ("calibration mismatch", dict(edit_dec=lambda d: d.update(calibration_method="isotonic" if d["calibration_method"] != "isotonic" else "sigmoid")), "decision_mismatch"),
    ("quick-labelled", dict(edit_meta=lambda m: m.update(training_mode="quick")), "quick_run_artifact"),
    ("legacy quick-like cfg", dict(strip_mode=True, edit_dec=lambda d: d["cfg"].update(hgb_iter=3, tune_rows=30000)), "quick_run_artifact"),
])
def test_loader_refuses_inconsistent_artifacts(art, tmp_path, name, kwargs, code):
    svc = ModelService.load(variant(art, tmp_path, **kwargs))
    assert not svc.loaded and svc.problem_code == code, name


def test_loader_quick_allowed_only_by_explicit_flag_and_legacy_full_is_inferred(art, tmp_path):
    d = variant(art, tmp_path / "q", edit_meta=lambda m: m.update(training_mode="quick"))
    assert ModelService.load(d, allow_quick=True).loaded
    legacy = ModelService.load(variant(art, tmp_path / "l", strip_mode=True))
    assert legacy.loaded and legacy.mode_inferred and legacy.training_mode == "full"


def test_loader_corrupt_model_and_missing_files(art, tmp_path):
    d = tmp_path / "bad"
    shutil.copytree(art, d)
    (d / "model.joblib").write_bytes(b"not a pickle")
    meta = json.loads((d / "model_meta.json").read_text())
    import hashlib
    meta["model_sha256"] = hashlib.sha256(b"not a pickle").hexdigest()
    (d / "model_meta.json").write_text(json.dumps(meta))
    assert ModelService.load(d).problem_code == "load_failed"
    assert ModelService.load(tmp_path / "nothing_here").problem_code == "artifacts_missing"


# ----------------------------------------------------------------------------- health
def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok", "api": "running", "model_loaded": True,
                                                  "model_name": r.json()["model_name"], "training_mode": "full", "problem_code": None}


def test_health_degraded_when_model_missing_and_predict_returns_safe_503(tmp_path):
    with TestClient(create_app(Settings(artifacts_dir=tmp_path / "SecretFolder" / "x")), raise_server_exceptions=False) as c:
        h = c.get("/health")
        assert h.status_code == 200 and h.json()["status"] == "degraded" and h.json()["model_loaded"] is False
        assert h.json()["problem_code"] == "artifacts_missing"
        r = c.post("/predict", json=_EXAMPLE)
        assert r.status_code == 503 and r.json()["error"] == "model_unavailable"
        assert "SecretFolder" not in r.text and "SecretFolder" not in h.text
        assert c.get("/model-info").status_code == 503


# ----------------------------------------------------------------------------- predictions
def test_valid_prediction_and_threshold_consistency(art, client):
    r = client.post("/predict", json=_EXAMPLE)
    assert r.status_code == 200
    b = r.json()
    dec = json.loads((art / "decisions_frozen.json").read_text())
    svc = client.app.state.service
    row = pd.DataFrame([[_EXAMPLE[n] for n in FEATURE_ORDER]], columns=FEATURE_ORDER).astype("float32")
    direct = float(svc.model.predict_proba_pos(row)[0])
    assert 0 <= b["estimated_probability"] <= 1 and abs(b["estimated_probability"] - direct) < 1e-6
    assert b["decision_threshold"] == dec["threshold"] == svc.model.threshold            # frozen, never recomputed
    assert b["above_threshold"] is (direct >= dec["threshold"])
    assert b["result_label"] == ("above_decision_threshold" if b["above_threshold"] else "below_decision_threshold")
    assert b["request_id"] == r.headers["x-request-id"]
    assert b["model"]["training_mode"] == "full" and b["model"]["calibration"] == dec["calibration_method"]


def test_prediction_is_deterministic_and_accepts_int_bmi(client):
    a, b = client.post("/predict", json=_EXAMPLE).json(), client.post("/predict", json=_EXAMPLE).json()
    assert a["estimated_probability"] == b["estimated_probability"]
    assert client.post("/predict", json={**_EXAMPLE, "bmi": 28}).status_code == 200


def test_threshold_boundary_uses_greater_or_equal(client):
    svc = client.app.state.service
    real, thr = svc.model, svc.model.threshold
    try:
        for p, expected in ((thr, True), (np.nextafter(thr, 0), False), (min(1.0, thr + 0.2), True), (0.0, False)):
            svc.model = StubModel(real, p=p)
            r = client.post("/predict", json=_EXAMPLE)
            assert r.status_code == 200 and r.json()["above_threshold"] is expected, p
            assert r.json()["decision_threshold"] == thr
    finally:
        svc.model = real


def test_model_info_exposes_only_safe_fields(client):
    r = client.get("/model-info")
    b = r.json()
    assert r.status_code == 200 and [f["name"] for f in b["features"]] == FEATURE_ORDER and len(b["features"]) == 19
    assert "existing" in b["target"].lower() and b["decision_threshold"] == client.app.state.service.model.threshold
    blob = r.text
    for leak in ("output_dir", "subset_file", "model_sha256", "\\\\", "/tmp", "/home", "Users"):
        assert leak not in blob


# ----------------------------------------------------------------------------- invalid / missing input
def test_all_inputs_missing_gives_19_field_errors(client):
    r = client.post("/predict", json={})
    assert r.status_code == 422 and r.json()["error"] == "validation_error"
    assert sorted(e["field"] for e in r.json()["errors"]) == sorted(FEATURE_ORDER)


@pytest.mark.parametrize("field", FEATURE_ORDER)
def test_each_single_missing_field_is_named(client, field):
    body = {k: v for k, v in _EXAMPLE.items() if k != field}
    r = client.post("/predict", json=body)
    assert r.status_code == 422 and [e["field"] for e in r.json()["errors"]] == [field]
    assert r.json()["errors"][0]["problem"] == "required field is missing"


@pytest.mark.parametrize("field,value", [
    ("age_group_5yr", 0), ("age_group_5yr", 14), ("age_group_5yr", "7"), ("age_group_5yr", 7.0), ("age_group_5yr", None), ("age_group_5yr", True),
    ("sex_female", 2), ("sex_female", -1), ("sex_female", False), ("bmi", 11.99), ("bmi", 100), ("bmi", "28"), ("bmi", None), ("bmi", True),
    ("gen_health", 6), ("phys_unhealthy_days", 31), ("phys_unhealthy_days", -1), ("phys_unhealthy_days", 88), ("ment_unhealthy_days", 77),
    ("diabetes_status", 5), ("diabetes_status", 7), ("smoker_status", 9), ("smoker_status", 0), ("education_level", 5), ("income_level", 8),
    ("income_level", [1]), ("hypertension_told", 3), ("heavy_drinker", "yes")])
def test_invalid_values_are_rejected_with_clear_error(client, field, value):
    r = client.post("/predict", json={**_EXAMPLE, field: value})
    assert r.status_code == 422
    errs = r.json()["errors"]
    assert [e["field"] for e in errs] == [field] and errs[0]["allowed"]
    assert str(value) not in json.dumps(errs) or value in (0, 1, True, False, None)       # submitted value is not echoed back


def test_unknown_field_nonjson_wrong_shape_and_empty_bodies(client):
    assert client.post("/predict", json={**_EXAMPLE, "is_smoker": 1}).json()["errors"][0]["field"] == "is_smoker"
    for kw in (dict(content=b"{not json", headers={"content-type": "application/json"}), dict(json=[1, 2]),
               dict(content=b"age=1", headers={"content-type": "text/plain"}), dict()):
        r = client.post("/predict", **kw)
        assert r.status_code == 422 and r.json()["errors"][0]["field"] == "body", kw


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_nan_and_infinity_literals_rejected(client, literal):
    body = json.dumps({**_EXAMPLE, "bmi": "@@"}).replace('"@@"', literal)
    r = client.post("/predict", content=body.encode(), headers={"content-type": "application/json"})
    assert r.status_code == 422 and r.json()["errors"][0]["field"] == "bmi"


def test_oversized_body_rejected_before_parsing(client):
    r = client.post("/predict", content=b" " * 20000, headers={"content-type": "application/json"})
    assert r.status_code == 413 and r.json()["error"] == "payload_too_large"


def test_http_errors_use_uniform_json_without_internals(client):
    for r, code in ((client.get("/does-not-exist"), 404), (client.get("/predict"), 405), (client.delete("/health"), 405)):
        assert r.status_code == code and set(r.json()) == {"error", "message", "request_id"}
    assert "Traceback" not in client.get("/does-not-exist").text


def test_request_id_echo_and_sanitisation(client):
    assert client.get("/health", headers={"X-Request-ID": "abc-12345678"}).headers["x-request-id"] == "abc-12345678"
    r = client.get("/health", headers={"X-Request-ID": "bad id!!<script>"})
    assert r.headers["x-request-id"] != "bad id!!<script>" and len(r.headers["x-request-id"]) == 32


# ----------------------------------------------------------------------------- safe error handling + logging
def test_internal_error_is_generic_and_logged_server_side(client, caplog):
    svc = client.app.state.service
    real = svc.model
    secret = r"C:\Users\akshat tiwari\SECRET\model_internal.py line 42"
    try:
        svc.model = StubModel(real, boom=RuntimeError(secret))
        with caplog.at_level("INFO"):
            r = client.post("/predict", json=_EXAMPLE)
    finally:
        svc.model = real
    assert r.status_code == 500 and r.json()["error"] == "internal_error" and set(r.json()) == {"error", "message", "request_id"}
    for leak in ("SECRET", "akshat", "C:\\", "Traceback", "RuntimeError", ".py", "line 42"):
        assert leak not in r.text and leak not in json.dumps(dict(r.headers)), leak
    assert r.headers["x-request-id"] == r.json()["request_id"]
    assert "RuntimeError" in caplog.text and "SECRET" in caplog.text                  # the detail IS available to operators


def test_server_recovers_after_internal_error(client):
    svc = client.app.state.service
    real = svc.model
    svc.model = StubModel(real, boom=ValueError("x"))
    assert client.post("/predict", json=_EXAMPLE).status_code == 500
    svc.model = real
    assert client.post("/predict", json=_EXAMPLE).status_code == 200


def test_invalid_probability_from_model_is_a_safe_500(client):
    svc = client.app.state.service
    real = svc.model
    try:
        svc.model = StubModel(real, p=float("nan"))
        r = client.post("/predict", json=_EXAMPLE)
    finally:
        svc.model = real
    assert r.status_code == 500 and r.json()["error"] == "internal_error"


def test_health_inputs_are_not_logged(client, caplog):
    with caplog.at_level("DEBUG"):
        client.post("/predict", json={**_EXAMPLE, "bmi": 33.337})
        client.post("/predict", json={**_EXAMPLE, "bmi": 4.4444})
    assert "33.337" not in caplog.text and "4.4444" not in caplog.text and "bmi" not in caplog.text.replace("fields=['bmi']", "")


def test_security_headers(client):
    r = client.post("/predict", json=_EXAMPLE)
    assert r.headers["cache-control"] == "no-store" and r.headers["x-content-type-options"] == "nosniff"
    assert "frame-ancestors 'none'" in r.headers["content-security-policy"]
    assert "content-security-policy" not in client.get("/docs").headers          # Swagger UI needs inline scripts


# ----------------------------------------------------------------------------- CORS / settings
def test_default_production_has_no_cors(art):
    s = Settings.from_env({})
    assert s.env == "production" and s.cors_origins == ()
    with TestClient(create_app(Settings(artifacts_dir=art)), raise_server_exceptions=False) as c:
        r = c.get("/health", headers={"Origin": "http://evil.example"})
        assert "access-control-allow-origin" not in r.headers
        r = c.options("/predict", headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"})
        assert "access-control-allow-origin" not in r.headers


def test_configured_origin_allowed_others_not_and_no_credentials(art):
    s = Settings.from_env({"API_CORS_ORIGINS": "https://app.example.org"})
    with TestClient(create_app(Settings(artifacts_dir=art, cors_origins=s.cors_origins)), raise_server_exceptions=False) as c:
        pre = c.options("/predict", headers={"Origin": "https://app.example.org", "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type"})
        assert pre.status_code == 200 and pre.headers["access-control-allow-origin"] == "https://app.example.org"
        assert "access-control-allow-credentials" not in pre.headers
        bad = c.options("/predict", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
        assert "access-control-allow-origin" not in bad.headers
        err = c.post("/predict", json={}, headers={"Origin": "https://app.example.org"})            # errors carry CORS headers too
        assert err.status_code == 422 and err.headers["access-control-allow-origin"] == "https://app.example.org"


def test_development_defaults_are_localhost_only():
    s = Settings.from_env({"API_ENV": "development"})
    assert s.cors_origins and all("localhost" in o or "127.0.0.1" in o for o in s.cors_origins)


@pytest.mark.parametrize("env", [{"API_CORS_ORIGINS": "*"}, {"API_CORS_ORIGINS": "https://a.org,*"}, {"API_ENV": "development", "API_CORS_ORIGINS": "*"},
                                 {"API_CORS_ORIGINS": "https://a.org/path"}, {"API_CORS_ORIGINS": "a.org"}, {"API_ENV": "staging"},
                                 {"API_LOG_LEVEL": "LOUD"}, {"API_MAX_BODY_BYTES": "10"}, {"API_ENABLE_DOCS": "maybe"}])
def test_bad_settings_fail_closed(env):
    with pytest.raises(ConfigError):
        create_app(Settings.from_env(env))


# ----------------------------------------------------------------------------- docs + honest wording
def test_swagger_and_openapi_available(client):
    assert client.get("/docs").status_code == 200 and "swagger" in client.get("/docs").text.lower()
    spec = client.get("/openapi.json").json()
    assert {"/predict", "/health", "/model-info"} <= set(spec["paths"])
    schema = spec["components"]["schemas"]["PredictionRequest"]
    assert list(schema["properties"]) == FEATURE_ORDER and schema["additionalProperties"] is False and set(schema["required"]) == set(FEATURE_ORDER)
    desc = spec["info"]["description"]
    for phrase in ("Educational/research prototype", "future heart attack", "India", "existing coronary heart disease"):
        assert phrase in desc


def test_docs_can_be_disabled(art):
    with TestClient(create_app(Settings(artifacts_dir=art, enable_docs=False)), raise_server_exceptions=False) as c:
        assert c.get("/docs").status_code == 404 and c.get("/openapi.json").status_code == 404 and c.get("/health").status_code == 200


def test_medical_wording_is_honest(client):
    b = client.post("/predict", json=_EXAMPLE).json()
    d = b["disclaimer"]
    for phrase in ("Educational/research prototype", "REPORT existing coronary heart disease", "CDC BRFSS 2025", "does NOT predict a future heart attack",
                   "does NOT diagnose", "does NOT establish clinical risk", "may not generalise to people in India"):
        assert phrase in d, phrase
    assert "existing" in b["target"].lower() and "not a diagnosis" in b["interpretation"].lower()
    assert "NOT reassurance" in text.INTERPRETATION_BELOW
    for banned in ("high risk", "high-risk", "you have heart disease", "you are safe", "you will have", "heart attack risk score"):
        assert banned not in (b["interpretation"] + d).lower(), banned


# ----------------------------------------------------------------------------- real artifacts: READ ONLY
def test_production_artifacts_load_readonly():
    if not (C.ARTIFACTS_DIR / "model.joblib").exists():
        pytest.skip("no production model present")
    svc = ModelService.load(C.ARTIFACTS_DIR)
    assert svc.loaded, f"production model not served: {svc.problem_code}"
    p, above = svc.predict(dict(_EXAMPLE))
    assert 0 <= p <= 1 and above == (p >= svc.model.threshold)
