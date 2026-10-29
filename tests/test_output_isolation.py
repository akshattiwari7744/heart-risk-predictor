"""Quick runs must never touch full-run (production) artifacts. Uses temp dirs only - never the real project."""
import hashlib
import json
from pathlib import Path

import joblib
import pytest

import train_model as T
from test_phase3 import make_frame

TINY = T.Cfg(cv_folds=2, hgb_iter=2, tune_rows=1500, n_boot=15, select_boot=15)
FILES = ["model.joblib", "metrics.json", "model_meta.json", "decisions_frozen.json"]
FIGS = ["val_pr_roc_candidates.png", "test_pr_roc_selected.png", "test_calibration.png", "test_confusion_matrix.png"]
DOCS = ["phase3_report.md", "model_card.md"]


def snapshot(root: Path, skip_dir: str = "quick"):
    """sha256 + mtime for every file NOT under a `quick` directory."""
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and skip_dir not in p.relative_to(root).parts:
            out[str(p.relative_to(root))] = (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
    return out


def run(env, *flags):
    return T.main(["--data-dir", str(env["data"]), "--artifacts-dir", str(env["art"]),
                   "--docs-dir", str(env["docs"]), *flags], cfg=TINY)


@pytest.fixture(scope="module")
def scenario(tmp_path_factory):
    root = tmp_path_factory.mktemp("proj")
    env = {"root": root, "data": root / "data", "art": root / "artifacts", "docs": root / "docs"}
    env["data"].mkdir()
    for name, (n, seed) in {"train": (3000, 1), "val": (1500, 2), "test": (1500, 3)}.items():
        make_frame(n, seed).to_parquet(env["data"] / f"{name}.parquet")
    run(env)                                              # 1. full run -> production paths
    full_snapshot = {k: v for k, v in snapshot(env["root"]).items() if not k.startswith("data")}
    run(env, "--quick")                                   # 2. quick run
    after_quick1 = {k: v for k, v in snapshot(env["root"]).items() if not k.startswith("data")}
    run(env, "--quick")                                   # 3. quick run again (overwrites only its own files)
    after_quick2 = {k: v for k, v in snapshot(env["root"]).items() if not k.startswith("data")}
    return {"env": env, "full": full_snapshot, "q1": after_quick1, "q2": after_quick2}


def test_paths_are_separate():
    full, quick = T.output_paths("full"), T.output_paths("quick")
    assert quick.artifacts != full.artifacts and quick.docs != full.docs and quick.figures != full.figures
    assert quick.artifacts == full.artifacts / "quick" and quick.docs == full.docs / "quick"
    assert full.artifacts.name == "artifacts" and full.docs.name == "docs"        # production paths unchanged
    with pytest.raises(ValueError):
        T.output_paths("fast")


def test_full_run_writes_production_files(scenario):
    e = scenario["env"]
    for f in FILES:
        assert (e["art"] / f).is_file(), f
    for f in FIGS:
        assert (e["art"] / "figures" / f).is_file(), f
    for f in DOCS:
        assert (e["docs"] / f).is_file(), f


def test_quick_run_writes_only_to_quick_dirs(scenario):
    e = scenario["env"]
    for f in FILES:
        assert (e["art"] / "quick" / f).is_file(), f
    for f in FIGS:
        assert (e["art"] / "quick" / "figures" / f).is_file(), f
    for f in DOCS:
        assert (e["docs"] / "quick" / f).is_file(), f


def test_quick_runs_never_modify_full_artifacts(scenario):
    # identical content AND identical mtime => not even rewritten with the same bytes
    assert scenario["q1"] == scenario["full"]
    assert scenario["q2"] == scenario["full"]


def test_training_mode_metadata(scenario):
    e = scenario["env"]
    for base, mode in ((e["art"], "full"), (e["art"] / "quick", "quick")):
        assert json.loads((base / "decisions_frozen.json").read_text())["training_mode"] == mode
        assert json.loads((base / "metrics.json").read_text())["training_mode"] == mode
        meta = json.loads((base / "model_meta.json").read_text())
        assert meta["training_mode"] == mode and Path(meta["output_dir"]) == base
        assert joblib.load(base / "model.joblib").meta["training_mode"] == mode
        assert json.loads((base / "decisions_frozen.json").read_text())["cfg"]["mode"] == mode


def test_quick_reports_are_labelled_and_full_are_not(scenario):
    e = scenario["env"]
    q = (e["docs"] / "quick" / "phase3_report.md").read_text(encoding="utf-8")
    f = (e["docs"] / "phase3_report.md").read_text(encoding="utf-8")
    assert "Training mode: QUICK" in q and "NOT final results" in q
    assert "QUICK SMOKE RUN" in (e["docs"] / "quick" / "model_card.md").read_text(encoding="utf-8")
    assert "Training mode: FULL" in f and "QUICK" not in f
    assert "EXISTING" in q and "EXISTING" in f              # target statement kept in both


def test_full_rerun_refused_without_force_and_files_unchanged(scenario):
    e = scenario["env"]
    before = snapshot(e["root"])
    with pytest.raises(SystemExit) as ex:
        run(e)
    assert "frozen" in str(ex.value).lower()
    assert snapshot(e["root"]) == before                   # refused BEFORE training; nothing written
    T.assert_safe_to_write(T.output_paths("full", e["art"], e["docs"]), e["art"], e["docs"], force=True)   # explicit opt-in works


def test_quick_pointed_at_production_dirs_is_refused(tmp_path):
    bad = T.OutPaths("quick", tmp_path / "artifacts", tmp_path / "artifacts" / "figures", tmp_path / "docs")
    with pytest.raises(RuntimeError):
        T.assert_safe_to_write(bad, tmp_path / "artifacts", tmp_path / "docs")


def test_quick_cfg_and_full_cfg_modes():
    assert T.quick_cfg().mode == "quick" and T.Cfg().mode == "full"
