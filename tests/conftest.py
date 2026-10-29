import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))            # so `import api` works with plain `pytest` too


def _snapshot():
    out = {}
    for sub in ("artifacts", "docs", "data/processed"):
        base = ROOT / sub
        if base.exists():
            for p in sorted(base.rglob("*")):
                if p.is_file() and "__pycache__" not in p.parts:
                    out[str(p.relative_to(ROOT))] = (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
    return out


@pytest.fixture(scope="session", autouse=True)
def production_files_unchanged():
    """Guard: no test may create, modify or delete production artifacts, docs or processed data."""
    before = _snapshot()
    yield
    after = _snapshot()
    changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    assert not changed, f"Tests modified production files: {changed}"
