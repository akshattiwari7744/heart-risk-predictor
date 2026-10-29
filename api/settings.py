"""Environment-driven settings. Secure by default: production mode, no cross-origin access, no wildcard ever."""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:           # config.py / modeling.py live in src/ (needed to unpickle FinalModel)
    sys.path.insert(0, str(SRC))

import config as C  # noqa: E402

DEV_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8443",
    "http://127.0.0.1:8443",
)
_ORIGIN_RE = re.compile(r"^https?://[A-Za-z0-9.-]+(:\d{1,5})?$")


class ConfigError(ValueError):
    pass


def _bool(name: str, raw: Optional[str], default: bool) -> bool:
    if raw is None or raw == "":
        return default
    v = raw.strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    raise ConfigError(f"{name} must be true or false")


@dataclass(frozen=True)
class Settings:
    env: str = "production"                      # production | development
    cors_origins: Tuple[str, ...] = ()
    artifacts_dir: Path = C.ARTIFACTS_DIR        # production artifacts (read-only use)
    log_level: str = "INFO"
    enable_docs: bool = True
    allow_quick_model: bool = False
    max_body_bytes: int = 16_384

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None) -> "Settings":
        e = os.environ if environ is None else environ
        env = (e.get("API_ENV") or "production").strip().lower()
        if env not in ("production", "development"):
            raise ConfigError("API_ENV must be 'production' or 'development'")
        raw = e.get("API_CORS_ORIGINS")
        if raw is None or raw.strip() == "":
            origins = DEV_ORIGINS if env == "development" else ()
        else:
            origins = tuple(o.strip() for o in raw.split(",") if o.strip())
        for o in origins:
            if o == "*" or not _ORIGIN_RE.match(o):
                raise ConfigError("API_CORS_ORIGINS must be a comma-separated list of explicit origins like "
                                  "https://example.org (no wildcard, no path, no trailing slash)")
        level = (e.get("API_LOG_LEVEL") or "INFO").strip().upper()
        if level not in ("DEBUG", "INFO", "WARNING", "ERROR"):
            raise ConfigError("API_LOG_LEVEL must be DEBUG, INFO, WARNING or ERROR")
        try:
            max_body = int(e.get("API_MAX_BODY_BYTES") or 16_384)
        except ValueError:
            raise ConfigError("API_MAX_BODY_BYTES must be an integer")
        if not 1_024 <= max_body <= 1_048_576:
            raise ConfigError("API_MAX_BODY_BYTES must be between 1024 and 1048576")
        return cls(env=env, cors_origins=origins,
                   artifacts_dir=Path(e["API_ARTIFACTS_DIR"]) if e.get("API_ARTIFACTS_DIR") else C.ARTIFACTS_DIR,
                   log_level=level, enable_docs=_bool("API_ENABLE_DOCS", e.get("API_ENABLE_DOCS"), True),
                   allow_quick_model=_bool("API_ALLOW_QUICK_MODEL", e.get("API_ALLOW_QUICK_MODEL"), False),
                   max_body_bytes=max_body)
