from pathlib import Path

import pytest

import make_lock as L

ROOT = Path(__file__).resolve().parents[1]


def test_parse_pins_and_missing():
    pins = L.parse_pins(["# c", "Scikit_Learn==1.5.0", "pandas==2.2.1", "pkg @ file:///x", ""])
    assert "scikit-learn" in pins and pins["pandas"] == "pandas==2.2.1" and pins["__other__"] == ["pkg @ file:///x"]
    assert L.missing_requirements(["pandas>=2.1", "scikit-learn>=1.4", "numpy>=1.26"], pins) == ["numpy"]


def test_existing_lock_pins_every_requirement():
    lock, req = ROOT / "requirements-lock.txt", ROOT / "requirements.txt"
    if not lock.exists():
        pytest.skip("requirements-lock.txt not created yet")
    pins = L.parse_pins(lock.read_text(encoding="utf-8").splitlines())
    assert L.missing_requirements(req.read_text().split(), pins) == []
