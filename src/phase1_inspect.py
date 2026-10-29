"""
Phase 1 - Heart Risk Predictor (BRFSS 2025)
-------------------------------------------
Scaffolds the project folders, reads LLCP2025.XPT exactly ONCE (read-only,
in chunks), verifies candidate columns, profiles values / special codes, and
writes docs. No model is trained. The XPT is never modified or copied.

Run from the project root:
    python src\\phase1_inspect.py
    python src\\phase1_inspect.py --xpt "D:\\path\\to\\LLCP2025.XPT"
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DIRS = [
    "data/raw", "data/interim", "data/processed", "docs", "notebooks", "src",
    "tests", "artifacts/figures", "api", "frontend",
]

TARGET = "_MICHD"
SRC_COLS = ("CVDINFR4", "CVDCRHD4")
WEIGHT = "_LLCPWT"
EXPECTED = {"rows": 356_158, "cols": 283, "target_missing": 4_013, "valid": 352_145}
SENTINELS = [7, 9, 77, 88, 99, 777, 888, 999, 7777, 9999, 77777, 99999, 777777, 999999]
MAX_LEVELS = 40
HIGH_MISSING_PCT = 20.0


def E(name, role, group="", handling="", note=""):
    return dict(name=name, role=role, group=group, handling=handling, note=note)


GEN = "7/9 -> NaN (DRAFT, confirm in codebook)"
SPEC = [
    # ---- candidate features (spec section 2) ----
    E("AGE", "feature", "age", "numeric; check range / top-coding in codebook"),
    E("_AGEG5YR", "feature_alt", "age", "ordinal 1-13; top code (14?) -> NaN (DRAFT, confirm)", "use if present; else AGE"),
    E("_AGE65YR", "feature_alt", "age", GEN, "2-group age; reference only"),
    E("_SEX", "feature", "sex", "categorical 1/2 (confirm)"),
    E("SEXVAR", "feature_alt", "sex", "categorical 1/2 (confirm)", "fallback for _SEX"),
    E("_BMI5", "feature", "", "divide by 100; NaN stays NaN; clip plausible range (DRAFT)"),
    E("GENHLTH", "feature", "", "ordinal 1-5; " + GEN),
    E("PHYSHLTH", "feature", "", "88 -> 0; 77/99 -> NaN (DRAFT, confirm)"),
    E("MENTHLTH", "feature", "", "88 -> 0; 77/99 -> NaN (DRAFT, confirm)"),
    E("_TOTINDA", "feature", "activity", "1/2; 9 -> NaN (DRAFT, confirm)"),
    E("EXERANY2", "feature_alt", "activity", GEN, "source of _TOTINDA; use one only"),
    E("BPHIGH6", "feature", "", "categorical, several yes/borderline levels; " + GEN),
    E("TOLDHI3", "feature", "", GEN),
    E("DIABETE4", "feature", "", "categorical; " + GEN, "name not confirmed for 2025"),
    E("ASTHMA3", "feature", "", GEN),
    E("CHCCOPD3", "feature", "", GEN),
    E("CHCKDNY2", "feature", "", GEN),
    E("ADDEPEV3", "feature", "", GEN),
    E("CHCOCNC1", "feature", "", GEN),
    E("_SMOKER3", "feature", "smoking", "categorical; 9 -> NaN (DRAFT, confirm)", "name not confirmed for 2025"),
    E("SMOKE100", "feature_alt", "smoking", GEN, "fallback if _SMOKER3 absent"),
    E("SMOKDAY2", "feature_alt", "smoking", GEN, "fallback if _SMOKER3 absent"),
    E("_RFDRHV*", "feature", "alcohol", "heavy-drinking flag; 9 -> NaN (DRAFT, confirm)", "suffix varies by year; matched by prefix"),
    E("_EDUCAG", "feature", "", "ordinal; 9 -> NaN (DRAFT, confirm)"),
    E("_INCOMG1", "feature", "", "ordinal; 9 -> NaN (DRAFT, confirm)", "was _INCOMG in earlier years"),
    E("_RACE", "feature_alt", "race", GEN, "optional; fairness-sensitive"),
    E("_IMPRACE", "feature_alt", "race", GEN, "optional; fairness-sensitive"),
    E("MARITAL", "feature", "", GEN, "optional"),
    E("EMPLOY1", "feature", "", GEN, "optional"),
    # ---- target / leakage / exclusions (spec section 3) ----
    E(TARGET, "target", "", "1 -> 1, 2 -> 0, anything else -> drop"),
    E("CVDINFR4", "excluded_leakage", "", "never a model input", "source of _MICHD"),
    E("CVDCRHD4", "excluded_leakage", "", "never a model input", "source of _MICHD"),
    E("CVDSTRK3", "excluded_related", "", "excluded v1", "related CV outcome; optional ablation only"),
    E("BPMEDS1", "excluded_related", "", "excluded v1", "treatment/diagnosis proxy"),
    E("CHOLMED3", "excluded_related", "", "excluded v1", "treatment/diagnosis proxy"),
    E("CHECKUP1", "excluded_access", "", "excluded v1", "healthcare-contact proxy"),
    E("PERSDOC3", "excluded_access", "", "excluded v1", "healthcare-contact proxy"),
    E("MEDCOST1", "excluded_access", "", "excluded v1", "healthcare-contact proxy"),
    E("_HLTHPL2", "excluded_access", "", "excluded v1", "healthcare-contact proxy"),
    E(WEIGHT, "weight_secondary", "", "not a feature; weighted metrics only"),
    E("_STSTR", "excluded_admin", "", "design variable"),
    E("_PSU", "excluded_admin", "", "design variable"),
    E("_STATE", "excluded_admin", "", "geography; excluded"),
    E("FMONTH", "excluded_admin", "", "administrative"),
    E("IDATE", "excluded_admin", "", "administrative"),
    E("IMONTH", "excluded_admin", "", "administrative"),
    E("IDAY", "excluded_admin", "", "administrative"),
    E("IYEAR", "excluded_admin", "", "administrative"),
    E("DISPCODE", "excluded_admin", "", "administrative"),
    E("SEQNO", "excluded_admin", "", "administrative"),
]
EXTRA_PATTERNS = ["_STRWT", "_RAWRAKE", "_WT2RAKE", "_LLCPWT2"]  # other weights, if present
LEAK_RE = re.compile(r"CVD|ASPR|ANGINA|HEART|CORON|_MICHD|STRK|STROKE|CHD", re.I)

REASONS = {
    "excluded_leakage": "Direct source of the target (_MICHD is derived from it). Must never be a feature.",
    "excluded_related": "Related cardiovascular outcome or treatment proxy; excluded from v1.",
    "excluded_access": "Healthcare-contact proxy for diagnosis; optional labelled experiment only.",
    "excluded_admin": "Survey design / administrative field; no valid predictive meaning.",
    "weight_secondary": "Survey weight; used only for secondary weighted evaluation.",
}


# --------------------------------------------------------------------------- helpers
def ensure_dirs() -> None:
    for d in DIRS:
        (ROOT / d).mkdir(parents=True, exist_ok=True)
    gi = ROOT / ".gitignore"
    if not gi.exists():
        gi.write_text("data/\n*.XPT\n*.xpt\n.venv/\n__pycache__/\n.ipynb_checkpoints/\nnode_modules/\n")


def find_xpt(arg: str | None) -> Path:
    if arg:
        p = Path(arg)
        if p.is_file():
            return p
        sys.exit(f"ERROR: file not found: {p}")
    for cand in (ROOT / "LLCP2025.XPT", ROOT / "data/raw/LLCP2025.XPT",
                 ROOT / "LLCP2025.xpt", ROOT / "data/raw/LLCP2025.xpt"):
        if cand.is_file():
            return cand
    sys.exit("ERROR: LLCP2025.XPT not found in project root or data/raw. Use --xpt <path>.")


def fmt(v) -> str:
    try:
        f = float(v)
        return str(int(f)) if f.is_integer() else f"{f:g}"
    except (TypeError, ValueError):
        return str(v)


def suggest(name: str, cols: list[str]) -> list[str]:
    base = name.rstrip("*")
    stem = re.sub(r"\d+$", "", base).upper()
    if len(stem) < 4:
        return []
    return [c for c in cols if c.upper().startswith(stem) and c.upper() != base.upper()][:8]


def resolve(entry: dict, cols: list[str]) -> list[str]:
    """Return actual column names matching a spec entry (exact, case-insensitive, or prefix*)."""
    n = entry["name"]
    if n.endswith("*"):
        return [c for c in cols if c.upper().startswith(n[:-1].upper())]
    hit = [c for c in cols if c == n]
    if not hit:
        hit = [c for c in cols if c.upper() == n.upper()]
    return hit


def clean_obj(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        return s.map(lambda b: b.decode("latin-1").strip() if isinstance(b, bytes) else b)
    return s


def profile(s: pd.Series) -> dict:
    n = len(s)
    nan = int(s.isna().sum())
    nn = s.dropna()
    d = {"dtype": str(s.dtype), "n": n, "nan": nan,
         "nan_pct": round(100 * nan / n, 2) if n else None,
         "n_unique": int(nn.nunique()), "min": None, "max": None,
         "levels": None, "sentinels": {}, "pctiles": None}
    if pd.api.types.is_numeric_dtype(nn) and len(nn):
        d["min"], d["max"] = fmt(nn.min()), fmt(nn.max())
        vc = nn.value_counts().sort_index()
        if len(vc) <= MAX_LEVELS:
            d["levels"] = {fmt(k): int(v) for k, v in vc.items()}
        else:
            q = nn.quantile([0.01, 0.5, 0.99])
            d["pctiles"] = {"p1": fmt(q.iloc[0]), "p50": fmt(q.iloc[1]), "p99": fmt(q.iloc[2])}
        d["sentinels"] = {fmt(c): int((nn == c).sum()) for c in SENTINELS if (nn == c).any()}
    elif len(nn):
        vc = nn.value_counts()
        d["levels"] = {str(k): int(v) for k, v in vc.head(MAX_LEVELS).items()}
    return d


def md_table(headers, rows) -> str:
    esc = lambda x: str(x).replace("|", "\\|")
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(esc(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# --------------------------------------------------------------------------- single-pass read
def read_once(xpt: Path, chunksize: int):
    """One sequential pass over the XPT. Keeps only needed columns in memory."""
    all_cols: list[str] = []
    nonnull = None
    keep: list[str] = []
    parts, total = [], 0
    t0 = time.time()
    with pd.read_sas(xpt, format="xport", chunksize=chunksize) as reader:
        nobs = getattr(reader, "nobs", None)
        for i, chunk in enumerate(reader):
            if i == 0:
                all_cols = list(chunk.columns)
                nonnull = pd.Series(0, index=all_cols, dtype="int64")
                want = set()
                for e in SPEC:
                    want.update(resolve(e, all_cols))
                for pat in EXTRA_PATTERNS:
                    want.update(c for c in all_cols if c.upper().startswith(pat.upper()))
                keep = [c for c in all_cols if c in want]
            nonnull = nonnull.add(chunk.notna().sum(), fill_value=0)
            sub = chunk[keep].copy()
            for c in keep:
                sub[c] = clean_obj(sub[c])
            parts.append(sub)
            total += len(chunk)
            el = time.time() - t0
            tot_txt = f"/{nobs:,}" if nobs else ""
            print(f"  read {total:,}{tot_txt} rows  ({el:,.0f}s)", end="\r", flush=True)
    print()
    return all_cols, nonnull.astype("int64"), pd.concat(parts, ignore_index=True), total


# --------------------------------------------------------------------------- analysis
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xpt", help="path to LLCP2025.XPT (default: project root or data/raw)")
    ap.add_argument("--chunksize", type=int, default=50_000)
    args = ap.parse_args()

    ensure_dirs()
    xpt = find_xpt(args.xpt)
    print(f"Project root : {ROOT}\nXPT (read-only): {xpt}  ({xpt.stat().st_size / 1e6:,.0f} MB)")
    print("Reading once in chunks; this can take several minutes...")
    t0 = time.time()
    all_cols, nonnull, df, n_rows = read_once(xpt, args.chunksize)
    n_cols = len(all_cols)

    report: list[str] = []          # console/report text
    problems: list[str] = []        # corrections needed

    # ---- integrity
    checks = [
        ("Rows in file", n_rows, EXPECTED["rows"]),
        ("Columns in file", n_cols, EXPECTED["cols"]),
    ]
    # ---- target
    has_target = TARGET in df.columns
    if has_target:
        t = df[TARGET]
        t_missing = int((~t.isin([1, 2])).sum())
        valid_mask = t.isin([1, 2])
        n_valid = int(valid_mask.sum())
        checks += [("Rows with missing/non-1-2 _MICHD", t_missing, EXPECTED["target_missing"]),
                   ("Rows after dropping missing target", n_valid, EXPECTED["valid"])]
    else:
        valid_mask = pd.Series(True, index=df.index)
        n_valid = len(df)
        problems.append(f"Target column {TARGET} NOT FOUND - stop and check column names.")
    integ_rows = [(k, f"{a:,}", f"{b:,}", "PASS" if a == b else "FAIL") for k, a, b in checks]
    for k, a, b, s in integ_rows:
        if s == "FAIL":
            problems.append(f"Integrity: {k} = {a}, expected {b}.")

    # ---- target details
    tgt_md: list[str] = []
    if has_target:
        vc_all = t.value_counts(dropna=False).sort_index()
        tgt_md.append(md_table(["_MICHD value (all rows)", "count"],
                               [("<NaN>" if pd.isna(k) else fmt(k), f"{v:,}") for k, v in vc_all.items()]))
        y = (t[valid_mask] == 1).astype(int)
        prev = y.mean()
        tgt_md.append(f"\nValid rows: **{n_valid:,}**; positives (=1): **{int(y.sum()):,}**; "
                      f"unweighted positive rate: **{100 * prev:.2f}%**.")
        if WEIGHT in df.columns:
            w = df.loc[valid_mask, WEIGHT]
            ok = w.notna()
            if ok.any():
                wp = np.average(y[ok], weights=w[ok])
                tgt_md.append(f"Weighted (`{WEIGHT}`) positive rate, informational only: {100 * wp:.2f}%.")
        if all(c in df.columns for c in SRC_COLS):
            a, b = df[SRC_COLS[0]], df[SRC_COLS[1]]
            der = pd.Series(np.nan, index=df.index)
            der[(a == 1) | (b == 1)] = 1
            der[(a == 2) & (b == 2)] = 2
            both = pd.DataFrame({"_MICHD": t, "re-derived": der})
            mism = int((both["_MICHD"].fillna(-1) != both["re-derived"].fillna(-1)).sum())
            tgt_md.append(f"\nRe-derivation check (1 if either source=1; 2 if both=2; else missing) vs `_MICHD`: "
                          f"**{mism:,} mismatching rows** (0 expected; non-zero means a rule/code difference - "
                          f"check codebook).")
            if mism:
                ct = pd.crosstab(both["_MICHD"].fillna(-1), both["re-derived"].fillna(-1))
                tgt_md.append("\n```\n" + ct.to_string() + "\n```")
                problems.append(f"_MICHD re-derivation mismatches: {mism:,} rows (see audit).")
        else:
            problems.append("CVDINFR4/CVDCRHD4 not both found; could not re-derive _MICHD.")

    # ---- per-variable verification
    dv = df[valid_mask]
    rows_spec, status_rows, vc_rows = [], [], []
    missing_vars, renamed = [], []
    profiles: dict[str, dict] = {}
    for e in SPEC:
        found = resolve(e, all_cols)
        if not found:
            sug = suggest(e["name"], all_cols)
            exact_needed = e["role"] in ("feature", "target") or e["role"].startswith("excluded_leak")
            missing_vars.append((e, sug))
            status_rows.append((e["name"], e["role"], "MISSING", "; ".join(sug) or "-", "-", "-"))
            rows_spec.append({
                "variable": e["name"], "role": e["role"], "model_input": "NO" if e["role"].startswith(("excluded", "target", "weight")) else ("alternate-not-found" if e["role"] == "feature_alt" else "candidate-not-found"),
                "in_dataset": "NO", "found_name": "", "suggested_alternatives": "|".join(sug),
                "dtype": "", "valid_rows_n": "", "nan_n": "", "nan_pct": "", "n_unique": "",
                "min": "", "max": "", "observed_values": "", "special_codes_observed": "",
                "draft_handling": e["handling"], "codebook_label": "", "codebook_confirmed": "NO",
                "notes": e["note"],
            })
            continue
        for col in found:
            p = profile(dv[col])
            profiles[col] = p
            renamed_flag = col != e["name"] and not e["name"].endswith("*")
            if renamed_flag:
                renamed.append((e["name"], col))
            mi = ("candidate" if e["role"] == "feature" else "alternate" if e["role"] == "feature_alt" else "NO")
            lv = ""
            if p["levels"] is not None:
                lv = "; ".join(f"{k}:{v}" for k, v in p["levels"].items())
                for k, v in p["levels"].items():
                    vc_rows.append((col, k, v, round(100 * v / max(p["n"], 1), 3)))
                if p["nan"]:
                    vc_rows.append((col, "<NaN>", p["nan"], round(100 * p["nan"] / p["n"], 3)))
            elif p["pctiles"]:
                lv = f"p1={p['pctiles']['p1']}, p50={p['pctiles']['p50']}, p99={p['pctiles']['p99']}"
            sent = "|".join(f"{k}:{v}" for k, v in p["sentinels"].items())
            rows_spec.append({
                "variable": e["name"], "role": e["role"], "model_input": mi,
                "in_dataset": "YES", "found_name": col, "suggested_alternatives": "",
                "dtype": p["dtype"], "valid_rows_n": p["n"], "nan_n": p["nan"], "nan_pct": p["nan_pct"],
                "n_unique": p["n_unique"], "min": p["min"], "max": p["max"],
                "observed_values": lv, "special_codes_observed": sent,
                "draft_handling": e["handling"], "codebook_label": "", "codebook_confirmed": "NO",
                "notes": e["note"] + (" | RENAMED/MATCHED BY PREFIX" if renamed_flag or e["name"].endswith("*") else ""),
            })
            status_rows.append((col, e["role"], "FOUND", "-", f"{p['nan_pct']}%", sent or "-"))

    # missing-variable corrections
    for e, sug in missing_vars:
        if e["role"] in ("feature", "target"):
            problems.append(f"Candidate/target `{e['name']}` NOT FOUND. "
                            + (f"Possible 2025 names: {', '.join(sug)}." if sug else "No similar column found."))
    # primaries that are absent but alternates exist, and vice-versa
    def present(n):
        return any(c.upper() == n.upper() for c in all_cols)
    for prim, alts in (("_AGEG5YR", ["AGE"]), ("_SEX", ["SEXVAR"]), ("_TOTINDA", ["EXERANY2"]),
                       ("_SMOKER3", ["SMOKE100", "SMOKDAY2"])):
        if not present(prim):
            av = [a for a in alts if present(a)]
            problems.append(f"`{prim}` absent -> use fallback: {', '.join(av) if av else 'NONE AVAILABLE'}.")
    if not (present("_RACE") or present("_IMPRACE")):
        problems.append("Neither `_RACE` nor `_IMPRACE` found (optional feature; skip or find the 2025 race variable).")
    if not any(c.upper().startswith("_RFDRHV") for c in all_cols):
        problems.append("No `_RFDRHV*` column found; heavy-drinking flag unavailable (check ALCDAY/AVEDRNK items).")

    # flags
    high_missing, constant = [], []
    for col, p in profiles.items():
        role = next((r["role"] for r in rows_spec if r["found_name"] == col), "")
        if role in ("feature", "feature_alt") and p["nan_pct"] is not None and p["nan_pct"] > HIGH_MISSING_PCT:
            high_missing.append((col, p["nan_pct"]))
        if p["n_unique"] <= 1:
            constant.append(col)
    for col, pct in high_missing:
        problems.append(f"`{col}` has {pct}% NaN among valid-target rows; decide: keep (+missing flag), skip-pattern "
                        f"category, or drop.")
    for col in constant:
        problems.append(f"`{col}` is constant among valid rows; drop.")
    if "_BMI5" in profiles and profiles["_BMI5"]["max"] is not None:
        problems.append(f"Check `_BMI5` scaling: observed min={profiles['_BMI5']['min']}, "
                        f"max={profiles['_BMI5']['max']} (expect values like 1200-7000 if stored x100).")
    for col, p in profiles.items():
        if col in ("PHYSHLTH", "MENTHLTH") and p["sentinels"]:
            problems.append(f"`{col}` special codes seen: {p['sentinels']} (confirm 88='none', 77/99=don't know/refused).")

    # leakage scan over ALL column names
    spec_names = {c for e in SPEC for c in resolve(e, all_cols)}
    leak_hits = [c for c in all_cols if LEAK_RE.search(c) and c not in spec_names]
    calc_vars = [c for c in all_cols if c.startswith("_") and c not in spec_names]
    # also list spec columns that matched the regex (informational)
    leak_known = [c for c in all_cols if LEAK_RE.search(c) and c in spec_names]

    # ---- write outputs
    docs = ROOT / "docs"
    spec_df = pd.DataFrame(rows_spec)
    spec_df.to_csv(docs / "feature_spec.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(vc_rows, columns=["variable", "value", "count", "pct_of_valid_rows"]).to_csv(
        docs / "value_counts.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"position": range(1, n_cols + 1), "column": all_cols,
                  "non_null": nonnull.reindex(all_cols).values,
                  "non_null_pct": (100 * nonnull.reindex(all_cols).values / max(n_rows, 1)).round(2)}
                 ).to_csv(docs / "all_columns.csv", index=False, encoding="utf-8-sig")
    interim = ROOT / "data/interim/brfss_phase1_subset.parquet"
    df.to_parquet(interim, index=False)

    elapsed = time.time() - t0
    md = [f"# Feature Audit - BRFSS 2025 (Phase 1)",
          f"_Generated {datetime.now():%Y-%m-%d %H:%M} by `src/phase1_inspect.py` in {elapsed:,.0f}s; "
          f"pandas {pd.__version__}. All numbers below come from your file; nothing is hand-entered._\n",
          "> Codes marked DRAFT are conventions, **not** confirmed. Check each against the 2025 codebook and set "
          "`codebook_confirmed` in `feature_spec.csv`.\n",
          "## 1. Dataset integrity",
          md_table(["Check", "Observed", "Expected", "Result"], integ_rows), "",
          f"XPT: `{xpt}` (read once, read-only). Profiles below use the **{n_valid:,}** rows with a valid target.\n",
          "## 2. Target `_MICHD`", *tgt_md, "",
          "## 3. Variable verification",
          md_table(["Column", "Role", "Status", "Similar names (if missing)", "NaN % (valid rows)", "Special codes seen"],
                   status_rows), ""]
    md += ["## 4. Missing or renamed variables"]
    if missing_vars:
        md.append(md_table(["Spec name", "Role", "Possible matches in file"],
                           [(e["name"], e["role"], ", ".join(s) or "none") for e, s in missing_vars]))
    else:
        md.append("None missing.")
    if renamed:
        md.append("\nCase-insensitive matches: " + ", ".join(f"{a} -> {b}" for a, b in renamed))
    md += ["", "## 5. Special-code findings (candidate features)",
           "Numeric codes from a standard BRFSS list that actually occur. Whether they mean "
           "'Don't know/Refused/Not asked' or a real answer must be confirmed per variable in the codebook. "
           "For continuous fields (`AGE`, `_BMI5`) these hits are probably genuine values, not codes.", ""]
    sc_rows = [(c, p["n_unique"], "; ".join(f"{k}:{v:,}" for k, v in p["sentinels"].items()) or "-")
               for c, p in profiles.items() if profiles[c]["sentinels"]]
    md.append(md_table(["Column", "Distinct values", "Sentinel-like codes (code:count)"], sc_rows) if sc_rows
              else "None detected.")
    md += ["", "Full value tables: `docs/value_counts.csv` (columns with <= 40 distinct values).",
           "", "## 6. Leakage audit",
           md_table(["Column", "Role", "Present", "Reason"],
                    [(e["name"], e["role"], "yes" if resolve(e, all_cols) else "no", REASONS[e["role"]])
                     for e in SPEC if e["role"] in REASONS]), "",
           "Hard rule: `_MICHD`, `CVDINFR4`, `CVDCRHD4` are never model inputs. The Phase 2 loader must assert this.", "",
           "### Name scan of all columns (needs your manual codebook review)",
           f"Matching /CVD|ASPR|ANGINA|HEART|CORON|STRK|STROKE|CHD/ and not in the spec: "
           f"{', '.join(leak_hits) if leak_hits else 'none'}.",
           f"Matching but already handled in the spec: {', '.join(leak_known) if leak_known else 'none'}.", "",
           f"### Calculated (`_`-prefixed) columns not in the spec ({len(calc_vars)})",
           "Review these for useful features or hidden leakage; do not add any without reading its definition:",
           "`" + "`, `".join(calc_vars) + "`" if calc_vars else "none", "",
           "## 7. Corrections needed",
           *(f"- {p}" for p in problems), *(["- None detected by automated checks."] if not problems else []),
           "- Manual: fill `codebook_label` and set `codebook_confirmed=YES` per variable after reading the codebook.",
           "", "## 8. Files written",
           "- `docs/feature_spec.csv`, `docs/feature_audit.md`, `docs/value_counts.csv`, `docs/all_columns.csv`, "
           "`docs/phase1_report.txt`",
           f"- `data/interim/brfss_phase1_subset.parquet` ({len(df):,} rows x {df.shape[1]} columns; "
           f"spec columns only, includes excluded/leakage columns for audit - Phase 2 must select by role)."]
    (docs / "feature_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # ---- console / report
    rep = ["=" * 64, "PHASE 1 REPORT", "=" * 64,
           *(f"{k}: {a} (expected {b}) -> {s}" for k, a, b, s in integ_rows)]
    if has_target:
        rep.append(f"Positive rate among valid rows: {100 * prev:.2f}%")
    rep.append(f"Candidate/alternate columns found: "
               f"{sum(1 for r in rows_spec if r['in_dataset']=='YES' and r['model_input'] in ('candidate','alternate'))}"
               f" ; spec entries missing: {len(missing_vars)}")
    rep.append("Missing: " + (", ".join(e['name'] for e, _ in missing_vars) or "none"))
    rep.append("Corrections needed:")
    rep += [f"  - {p}" for p in problems] or ["  - none detected"]
    rep.append(f"Name-scan hits to review manually: {', '.join(leak_hits) or 'none'}")
    rep.append(f"Total runtime: {elapsed:,.0f}s")
    text = "\n".join(rep)
    (docs / "phase1_report.txt").write_text(text + "\n", encoding="utf-8")
    print("\n" + text)
    print(f"\nWrote files under {docs} and {interim}")


if __name__ == "__main__":
    main()
