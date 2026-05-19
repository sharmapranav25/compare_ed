"""Per-vendor accuracy vs hand-verified ground truth.

Walks tests/golden/*.json (and optionally tests/holdout/*.json), finds the
corresponding cached `.v2.cards.json` sidecar, and computes per-field accuracy:
for each verified golden SKU, does the extracted value match the verified value?

This is the GOLD STANDARD accuracy metric — strictly stronger than the
oracle-pass-rate or judge-agreement proxies we use elsewhere. Run before
shipping any pipeline change to validate true accuracy didn't regress.

Output: per-vendor table + aggregate. Exit code 1 if accuracy regressed below
the locked baseline in tests/golden_baseline.json.

Usage:
    # Score against current cached sidecars (in ~/buysheet_runs/ or files/)
    python -m buysheet_v2.tools.eval_against_goldens

    # Update baseline after a verified improvement
    python -m buysheet_v2.tools.eval_against_goldens --update-baseline

    # Include holdout vendors (the cold-vendor ship gate)
    python -m buysheet_v2.tools.eval_against_goldens --include-holdout
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from buysheet_v2.consistency import (
    deterministic_fill, detect_catalog_brand,
    drop_invalid_sku_cards, normalize_extraction,
)
from buysheet_v2.schemas.extraction_result import CatalogExtraction
from buysheet_v2.verify import verify_catalog

REPO = Path(__file__).resolve().parent.parent.parent
GOLDEN_DIR = REPO / "buysheet_v2" / "tests" / "golden"
HOLDOUT_DIR = REPO / "buysheet_v2" / "tests" / "holdout"
BASELINE_PATH = REPO / "buysheet_v2" / "tests" / "golden_baseline.json"
RUNS_DIR = Path.home() / "buysheet_runs"
DEFAULT_TOLERANCE_PP = 0.5

FIELDS_TO_EVAL = [
    "sku", "brand", "description", "color", "standard_color",
    "mg", "sg", "ssg", "intro_date", "usd_cost", "usd_retail",
]


def _normalize(value) -> str:
    if value is None:
        return ""
    s = str(value).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def find_sidecar_for(golden: dict) -> Optional[Path]:
    """Locate the most recent v2 sidecar matching this golden's vendor."""
    # Prefer the source_sidecar explicitly referenced
    src = golden.get("source_sidecar")
    if src and Path(src).exists():
        return Path(src)
    vendor_key = golden.get("vendor_key", "")
    # Search ~/buysheet_runs/ for any sidecar whose parent name approximates
    candidates: list[Path] = []
    if RUNS_DIR.exists():
        for vendor_dir in RUNS_DIR.iterdir():
            if not vendor_dir.is_dir():
                continue
            if vendor_key not in vendor_dir.name and vendor_dir.name not in vendor_key:
                continue
            for run_dir in sorted(vendor_dir.iterdir(), reverse=True):
                if not run_dir.is_dir():
                    continue
                for sc in run_dir.glob("*.v2.cards.json"):
                    candidates.append(sc)
    # Also check files/<vendor>/
    files_dir = REPO / "files"
    if files_dir.exists():
        for vendor_dir in files_dir.iterdir():
            if not vendor_dir.is_dir():
                continue
            if vendor_key not in vendor_dir.name and vendor_dir.name not in vendor_key:
                continue
            for sc in vendor_dir.glob("*.v2.cards.json"):
                candidates.append(sc)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def score_vendor(golden: dict, sidecar: Path) -> dict:
    """Score per-field accuracy for one vendor."""
    ext = CatalogExtraction.model_validate_json(sidecar.read_text())
    for pe in ext.pages:
        survivors, _ = drop_invalid_sku_cards(pe.cards)
        pe.cards = survivors
    normalize_extraction(ext.all_cards)
    page_text_by = {p.page: p.page_text or "" for p in ext.pages}
    deterministic_fill(ext.all_cards, page_text_by,
                       catalog_brand=detect_catalog_brand(ext.all_cards))
    ext = verify_catalog(ext)

    extracted_by_sku = {c.sku: c for c in ext.all_cards}

    per_field: dict[str, dict[str, int]] = {
        f: {"compared": 0, "match": 0, "mismatch": 0,
            "extracted_null": 0, "golden_null": 0}
        for f in FIELDS_TO_EVAL
    }
    skus_compared = 0
    skus_missed = 0  # in golden but not extracted
    mismatch_samples: list = []

    for gentry in golden.get("skus", []):
        if not gentry.get("_verified"):
            continue
        sku = gentry.get("sku")
        if not sku:
            continue
        card = extracted_by_sku.get(sku)
        if card is None:
            skus_missed += 1
            continue
        skus_compared += 1
        for f in FIELDS_TO_EVAL:
            g_val = gentry.get(f)
            e_val = getattr(card, f, None)
            if g_val is None and e_val is None:
                continue
            per_field[f]["compared"] += 1
            if g_val is None:
                per_field[f]["golden_null"] += 1
                continue
            if e_val is None:
                per_field[f]["extracted_null"] += 1
                if len(mismatch_samples) < 12:
                    mismatch_samples.append((sku, f, g_val, e_val))
                continue
            if _normalize(g_val) == _normalize(e_val):
                per_field[f]["match"] += 1
            else:
                per_field[f]["mismatch"] += 1
                if len(mismatch_samples) < 12:
                    mismatch_samples.append((sku, f, g_val, e_val))

    total_compared = sum(pf["compared"] for pf in per_field.values())
    total_match = sum(pf["match"] for pf in per_field.values())
    overall = 100 * total_match / total_compared if total_compared else 0.0

    return {
        "skus_compared": skus_compared,
        "skus_missed_from_extraction": skus_missed,
        "total_field_comparisons": total_compared,
        "total_field_matches": total_match,
        "overall_accuracy_pct": overall,
        "per_field": per_field,
        "mismatch_samples": mismatch_samples,
    }


def load_baseline() -> dict:
    if BASELINE_PATH.exists():
        try:
            return json.loads(BASELINE_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {"vendors": {}, "_meta": {}}


def save_baseline(baseline: dict) -> None:
    baseline.setdefault("_meta", {})
    baseline["_meta"]["last_updated"] = datetime.now(timezone.utc).isoformat()
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_PATH.write_text(json.dumps(baseline, indent=2, sort_keys=True))


def main() -> int:
    ap = argparse.ArgumentParser(prog="eval_against_goldens")
    ap.add_argument("--include-holdout", action="store_true",
                    help="Also evaluate tests/holdout/*.json (cold-vendor ship gate)")
    ap.add_argument("--update-baseline", action="store_true",
                    help="Write current measurements into golden_baseline.json")
    ap.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE_PP)
    ap.add_argument("--vendor", default=None,
                    help="Evaluate only this vendor key (default: all)")
    args = ap.parse_args()

    golden_files = sorted(GOLDEN_DIR.glob("*.json"))
    if args.include_holdout:
        golden_files += sorted(HOLDOUT_DIR.glob("*.json"))
    if args.vendor:
        golden_files = [p for p in golden_files if p.stem == args.vendor]

    if not golden_files:
        print(f"no golden files found", file=sys.stderr)
        return 2

    baseline = load_baseline()
    baseline_vendors = baseline.get("vendors", {})
    any_regressed = False
    current = {}

    print(f"\n{'='*88}")
    print(f"GOLDEN EVAL — accuracy vs hand-verified ground truth")
    print('='*88)
    print(f"{'vendor':<28}  {'verified':>8}  {'compared':>8}  {'accuracy':>9}  {'Δ vs base':>10}  status")
    print('-'*88)

    for gpath in golden_files:
        golden = json.loads(gpath.read_text())
        vendor_key = golden.get("vendor_key", gpath.stem)
        verified = sum(1 for s in golden.get("skus", []) if s.get("_verified"))
        if verified == 0:
            print(f"  {vendor_key[:28]:<28}  {0:>8}  {'(skip)':>8}  no verified SKUs yet")
            continue
        sidecar = find_sidecar_for(golden)
        if sidecar is None:
            print(f"  {vendor_key[:28]:<28}  {verified:>8}  (no sidecar found in ~/buysheet_runs/ or files/)")
            continue
        result = score_vendor(golden, sidecar)
        current[vendor_key] = {
            "overall_accuracy_pct": result["overall_accuracy_pct"],
            "skus_compared": result["skus_compared"],
            "skus_missed_from_extraction": result["skus_missed_from_extraction"],
            "per_field_match_rate": {
                f: (100 * pf["match"] / pf["compared"]) if pf["compared"] else None
                for f, pf in result["per_field"].items()
            },
        }
        base_pct = baseline_vendors.get(vendor_key, {}).get("overall_accuracy_pct")
        delta_str = ""
        status = ""
        if base_pct is not None:
            delta = result["overall_accuracy_pct"] - base_pct
            delta_str = f"{delta:+.2f}pp"
            if delta < -args.tolerance:
                status = "⚠ REGRESSED"
                any_regressed = True
            elif delta > args.tolerance:
                status = "✓ improved"
        else:
            delta_str = "(NEW)"
        print(f"  {vendor_key[:28]:<28}  {verified:>8}  {result['skus_compared']:>8}  "
              f"{result['overall_accuracy_pct']:>7.2f}%   {delta_str:>10}  {status}")

    print()
    if not current:
        print("no verified goldens to score. Run verify_golden.py first.")
        return 0

    # Per-field detail
    print(f"\n{'='*88}")
    print(f"PER-FIELD MATCH RATE BREAKDOWN")
    print('='*88)
    header = f"{'field':<18}" + "".join(f"  {v[:14]:>14}" for v in current)
    print(header)
    print("-" * len(header))
    for f in FIELDS_TO_EVAL:
        row = f"{f:<18}"
        for vk, cd in current.items():
            rate = cd["per_field_match_rate"].get(f)
            row += f"  {('n/a' if rate is None else f'{rate:.1f}%'):>14}"
        print(row)

    if args.update_baseline:
        baseline["vendors"] = current
        save_baseline(baseline)
        print(f"\n[baseline updated] -> {BASELINE_PATH}")
        return 0

    if any_regressed:
        print(f"\n[REGRESSION DETECTED] tolerance was {args.tolerance}pp", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
