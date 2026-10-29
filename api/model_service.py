"""Loads the frozen Phase 3 model ONCE and refuses to serve it unless it agrees with config + metadata. Never trains or refits."""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from api.schemas import _EXAMPLE, FEATURE_ORDER, FIELD_HELP
from api.settings import C

log = logging.getLogger("heart_api.model")


def infer_mode(decisions: dict, meta: dict) -> Tuple[str, bool]:
    """('full'|'quick', inferred?). Pre-isolation artifacts have no training_mode: infer from the stored cfg."""
    for src in (meta.get("training_mode"), decisions.get("training_mode"), (decisions.get("cfg") or {}).get("mode")):
        if src in ("full", "quick"):
            return src, False
    cfg = decisions.get("cfg") or {}
    return ("quick" if (cfg.get("hgb_iter") == 3 and cfg.get("tune_rows") == 30000) else "full"), True


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class ModelService:
    artifacts_dir: Path
    model: object = None
    meta: dict = field(default_factory=dict)
    decisions: dict = field(default_factory=dict)
    training_mode: Optional[str] = None
    mode_inferred: bool = False
    problem_code: Optional[str] = None

    @property
    def loaded(self) -> bool:
        return self.model is not None and self.problem_code is None

    def _fail(self, code: str, *, exc: bool = False) -> "ModelService":
        self.model, self.problem_code = None, code
        (log.exception if exc else log.error)("model not loaded: %s (dir=%s)", code, self.artifacts_dir)   # path only in server log
        return self

    @classmethod
    def load(cls, artifacts_dir, allow_quick: bool = False) -> "ModelService":
        svc = cls(artifacts_dir=Path(artifacts_dir))
        d = svc.artifacts_dir
        mp, dp, metap = d / "model.joblib", d / "decisions_frozen.json", d / "model_meta.json"
        if not (mp.is_file() and dp.is_file() and metap.is_file()):
            return svc._fail("artifacts_missing")
        try:
            svc.decisions = json.loads(dp.read_text(encoding="utf-8"))
            svc.meta = json.loads(metap.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return svc._fail("metadata_unreadable", exc=True)
        svc.training_mode, svc.mode_inferred = infer_mode(svc.decisions, svc.meta)
        if svc.training_mode == "quick" and not allow_quick:
            return svc._fail("quick_run_artifact")
        if _sha256(mp) != svc.meta.get("model_sha256"):
            return svc._fail("integrity_check_failed")
        import pandas, sklearn
        saved = svc.meta.get("versions", {})
        if saved.get("scikit-learn") != sklearn.__version__:
            log.error("scikit-learn version differs from training (saved %s, installed %s)", saved.get("scikit-learn"), sklearn.__version__)
            return svc._fail("version_mismatch")
        for lib, now in (("pandas", pandas.__version__), ("numpy", np.__version__)):
            if saved.get(lib) != now:
                log.warning("%s version differs from training (saved %s, installed %s)", lib, saved.get(lib), now)
        try:
            from modeling import FinalModel
            model = joblib.load(mp)                       # trusted local artifact (hash-checked above); never load untrusted pickles
        except Exception:
            return svc._fail("load_failed", exc=True)
        cfg_features = [f["out"] for f in C.FEATURES]
        if not isinstance(model, FinalModel):
            return svc._fail("unexpected_model_type")
        if not (list(model.feature_names) == cfg_features == list(svc.meta.get("features", [])) == FEATURE_ORDER):
            return svc._fail("feature_mismatch")
        dec_thr = svc.decisions.get("threshold")
        if not (model.threshold == dec_thr == svc.meta.get("threshold")):
            return svc._fail("threshold_mismatch")
        if not (model.model_name == svc.decisions.get("selected_model") and
                getattr(model.calibrator, "method", None) == svc.decisions.get("calibration_method")):
            return svc._fail("decision_mismatch")
        svc.model = model
        try:                                              # self-test: one prediction; refits nothing
            svc.predict(dict(_EXAMPLE))
        except Exception:
            return svc._fail("self_test_failed", exc=True)
        log.info("model loaded: %s | calibration=%s | mode=%s%s | threshold=%.6f", model.model_name, model.calibrator.method,
                 svc.training_mode, " (inferred)" if svc.mode_inferred else "", model.threshold)
        return svc

    def predict(self, values: dict) -> Tuple[float, bool]:
        names = list(self.model.feature_names)
        row = pd.DataFrame([[values[n] for n in names]], columns=names).astype("float32")     # same dtype as training data
        p = float(self.model.predict_proba_pos(row)[0])
        if not np.isfinite(p) or p < 0.0 or p > 1.0:
            raise RuntimeError("model returned an invalid probability")
        return p, bool(p >= self.model.threshold)                                             # frozen threshold, never recomputed

    def features_info(self):
        return [{"name": n, **FIELD_HELP[n]} for n in self.model.feature_names]
