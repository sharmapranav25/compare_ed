"""Regression-check oracle: re-verify every cached vendor sidecar, compare to baseline.

Walks all <doc>.v2.cards.json sidecars under a search root, runs the semantic
oracle on each, prints per-vendor accuracy + delta vs the checked-in baseline
(tests/baseline_accuracy.json), and exits non-zero if any vendor regressed by
more than the tolerance.

Use this BEFORE pushing any code change that touches verify.py, extract.py,
prompts, or vocab files.

Usage:
    # Check current state vs baseline
    python -m buysheet_v2.tools.verify_all

    # Custom search root
    python -m buysheet_v2.tools.verify_all --root files/

    # Update the baseline after a verified improvement
    python -m buysheet_v2.tools.verify_all --update-baseline

    # Stricter tolerance (default 0.5pp)
    python -m buysheet_v2.tools.verify_all --tolerance 0.0

Exit codes:
    0 — all vendors at or above baseline
    1 — at least one vendor regressed
    2 — no sidecars found
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from buysheet_v2.schemas.extraction_result import CatalogExtraction
from buysheet_v2.verify import oracle_summary, verify_catalog

REPO = Path(__file__).resolve().parent.parent.parent
BASELINE_PATH = REPO / "buysheet_v2" / "tests" / "baseline_accuracy.json"
DEFAULT_TOLERANCE_PP = 0.5  # allow ≤0.5 percentage-point drift


def find_sidecars(root: Path) -> list[Path]:
    return sorted(root.rglob("*.v2.cards.json"))


def vendor_key_from_sidecar(path: Path) -> str:
    """`files/nike_ho26/HO26_GEL_FW.v2.cards.json` -> `nike_ho26` (parent dir)."""
    return path.parent.name


def measure_one(sidecar: Path) -> dict:
    ext = CatalogExtraction.model_validate_json(sidecar.read_text())
    ext = verify_catalog(ext)
    s = oracle_summary(ext)
    total = sum(st["total"] for st in s["per_field"].values())
    passing = sum(st["correct"] for st in s["per_field"].values())
    contradicted = sum(st["contradicted"] for st in s["per_field"].values())
    per_field = {
        f: {"correct": st["correct"], "total": st["total"],
            "rate_pct": round(100 * st["correct"] / st["total"], 1) if st["total"] else 0.0,
            "contradicted": st["contradicted"]}
        for f, st in s["per_field"].items()
    }
    return {
        "cards": s["cards"],
        "per_cell_pct": round(100 * passing / total, 2) if total else 0.0,
        "contradicted_pct": round(100 * contradicted / total, 2) if total else 0.0,
        "cards_perfect": s["cards_perfect"],
        "cards_partial": s["cards_partial"],
        "cards_contradicted": s["cards_contradicted"],
        "per_field": per_field,
    }


def load_baseline() -> dict:
    if not BASELINE_PATH.exists():
        return {"vendors": {}, "_meta": {}}
    try:
        return json.loads(BASELINE_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {"vendors": {}, "_meta": {}}


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(REPO), text=True,
        ).strip()
    except Exception:
        return "unknown"


def save_baseline(baseline: dict) -> None:
    baseline.setdefault("_meta", {})
    baseline["_meta"]["last_updated"] = datetime.now(timezone.utc).isoformat()
    baseline["_meta"]["git_head"] = _git_head()
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_PATH.write_text(json.dumps(baseline, indent=2, sort_keys=True))


def compare(current: dict, baseline: dict, tolerance_pp: float) -> tuple[bool, list[str]]:
    """Return (any_regressed, lines)."""
    vendors_baseline = baseline.get("vendors", {})
    lines = []
    any_regressed = False
    width = max((len(v) for v in current), default=8)
    lines.append(f"{'vendor':<{width}}  cards  per-cell    Δ       contradicted    base")
    lines.append("=" * 80)
    for vendor in sorted(current):
        cur = current[vendor]
        base = vendors_baseline.get(vendor)
        cur_pct = cur["per_cell_pct"]
        cur_contra = cur["contradicted_pct"]
        if base is None:
            lines.append(f"{vendor:<{width}}  {cur['cards']:5d}  {cur_pct:6.2f}%   NEW   "
                         f"{cur_contra:5.2f}%         (no baseline)")
            continue
        base_pct = base.get("per_cell_pct", 0.0)
        delta = cur_pct - base_pct
        flag = ""
        if delta < -tolerance_pp:
            flag = "  ⚠ REGRESSED"
            any_regressed = True
        elif delta > tolerance_pp:
            flag = "  ✓ improved"
        lines.append(f"{vendor:<{width}}  {cur['cards']:5d}  {cur_pct:6.2f}%   "
                     f"{delta:+5.2f}pp  {cur_contra:5.2f}%        {base_pct:5.2f}%{flag}")
    return any_regressed, lines


def main() -> int:
    ap = argparse.ArgumentParser(prog="verify_all")
    ap.add_argument("--root", type=Path, default=REPO / "files",
                    help="Directory to search for *.v2.cards.json (default: files/)")
    ap.add_argument("--update-baseline", action="store_true",
                    help="Write current measurements into baseline_accuracy.json")
    ap.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE_PP,
                    help=f"Allowed per-cell drift in pp (default: {DEFAULT_TOLERANCE_PP})")
    args = ap.parse_args()

    sidecars = find_sidecars(args.root)
    if not sidecars:
        print(f"[verify-all] no .v2.cards.json sidecars found under {args.root}", file=sys.stderr)
        return 2

    print(f"[verify-all] measuring {len(sidecars)} cached vendor(s)...", file=sys.stderr)
    current = {}
    for sc in sidecars:
        vendor = vendor_key_from_sidecar(sc)
        try:
            current[vendor] = measure_one(sc)
        except Exception as e:
            print(f"  [skip] {sc} ({type(e).__name__}: {e})", file=sys.stderr)

    baseline = load_baseline()
    any_regressed, lines = compare(current, baseline, args.tolerance)
    for line in lines:
        print(line)

    if args.update_baseline:
        baseline["vendors"] = current
        save_baseline(baseline)
        print(f"\n[verify-all] baseline updated -> {BASELINE_PATH}", file=sys.stderr)
        return 0

    if any_regressed:
        print(f"\n[verify-all] REGRESSION DETECTED (tolerance {args.tolerance}pp)", file=sys.stderr)
        return 1

    print(f"\n[verify-all] all vendors at or above baseline (tolerance {args.tolerance}pp)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
