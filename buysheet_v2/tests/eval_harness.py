"""Per-vendor accuracy harness.

Reads a golden test set (hand-verified ground truth, one entry per SKU),
runs the pipeline on the source PDF, and computes per-field accuracy.

Golden JSON shape:
{
  "vendor_key": "adidas_premium",
  "pdf_path": "files/adidas_premium/FW26_ORIGINALS_FTW_PREMIUM_RANGE.pdf",
  "description": "Hand-verified 20 SKUs from FW26 ORIGINALS FTW PREMIUM RANGE",
  "verified_at": "2026-05-17",
  "verified_by": "user",
  "skus": [
    {
      "sku": "KH7629",
      "page": 4,
      "brand": "adidas",
      "description": "FORUM SQ TRAINER W",
      "color": "cream / white/ivory/GUM 2",
      "standard_color": "White",
      "mg": "W-Footwear",
      "sg": "Sneakers",
      "ssg": "Court",
      "intro_date": "JUL",
      "usd_cost": 130.00,
      "usd_retail": null,
      "has_photo": true
    },
    ...
  ]
}

Ship gate (Phase 4): >=85% per-cell accuracy AND >=90% per-card accuracy on
holdout vendors (tests/holdout/*.json).
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

FIELDS_TO_EVAL = [
    "brand", "description", "color", "standard_color",
    "mg", "sg", "ssg", "intro_date", "usd_cost", "usd_retail",
]


def load_golden(path: Path) -> dict:
    return json.loads(path.read_text())


def compute_accuracy(golden: dict, extracted_cards: list[dict]) -> dict:
    """Compare extracted cards to golden ground-truth, return per-field accuracy.

    Cards are matched by SKU. A card with no matching golden SKU is "extra".
    A golden SKU with no extracted card is "missed".
    """
    extracted_by_sku = {c["sku"]: c for c in extracted_cards}
    golden_by_sku = {s["sku"]: s for s in golden["skus"]}

    per_field_correct = Counter()
    per_field_total = Counter()
    cards_perfect = 0
    cards_partial = 0
    cards_missed = 0
    extra_cards = 0

    for sku, truth in golden_by_sku.items():
        extracted = extracted_by_sku.get(sku)
        if extracted is None:
            cards_missed += 1
            for f in FIELDS_TO_EVAL:
                if truth.get(f) is not None:
                    per_field_total[f] += 1
            continue
        card_ok = True
        for f in FIELDS_TO_EVAL:
            truth_v = truth.get(f)
            if truth_v is None:
                continue
            per_field_total[f] += 1
            ext_v = extracted.get(f)
            if _values_match(f, truth_v, ext_v):
                per_field_correct[f] += 1
            else:
                card_ok = False
        if card_ok:
            cards_perfect += 1
        else:
            cards_partial += 1

    extra_cards = sum(1 for sku in extracted_by_sku if sku not in golden_by_sku)

    total_cells = sum(per_field_total.values())
    correct_cells = sum(per_field_correct.values())
    return {
        "vendor": golden.get("vendor_key"),
        "golden_sku_count": len(golden_by_sku),
        "extracted_sku_count": len(extracted_by_sku),
        "extra_cards": extra_cards,
        "cards_perfect": cards_perfect,
        "cards_partial": cards_partial,
        "cards_missed": cards_missed,
        "per_field": {
            f: {
                "correct": per_field_correct[f],
                "total": per_field_total[f],
                "rate": (per_field_correct[f] / per_field_total[f])
                if per_field_total[f] else None,
            }
            for f in FIELDS_TO_EVAL
        },
        "per_cell_accuracy": correct_cells / total_cells if total_cells else None,
        "per_card_accuracy": cards_perfect / len(golden_by_sku) if golden_by_sku else None,
    }


def _values_match(field: str, truth, extracted) -> bool:
    """Field-aware equality check."""
    if truth is None and extracted is None:
        return True
    if truth is None or extracted is None:
        return False
    if field in ("usd_cost", "usd_retail"):
        try:
            return abs(float(truth) - float(extracted)) < 0.01
        except (TypeError, ValueError):
            return False
    return str(truth).strip().lower() == str(extracted).strip().lower()


def format_report(result: dict) -> str:
    lines = []
    lines.append(f"=== Eval: {result['vendor']} ===")
    lines.append(f"  golden SKUs: {result['golden_sku_count']}  extracted: {result['extracted_sku_count']}  extra: {result['extra_cards']}")
    lines.append(f"  cards perfect: {result['cards_perfect']}  partial: {result['cards_partial']}  missed: {result['cards_missed']}")
    pc = result["per_card_accuracy"]
    pca = result["per_cell_accuracy"]
    lines.append(f"  per-card accuracy: {(pc or 0):.1%}   per-cell accuracy: {(pca or 0):.1%}")
    lines.append("  per-field:")
    for f, stats in result["per_field"].items():
        rate = stats["rate"]
        rate_s = f"{rate:.1%}" if rate is not None else "n/a"
        lines.append(f"    {f:18s}  {stats['correct']:3d}/{stats['total']:3d}  ({rate_s})")
    return "\n".join(lines)


def run_eval(vendor: str, golden_path: Path) -> int:
    """Run pipeline on golden's PDF + compute accuracy. Returns process exit code."""
    print(f"[eval] loading golden: {golden_path}", file=sys.stderr)
    golden = load_golden(golden_path)
    pdf_path = Path(golden["pdf_path"])
    if not pdf_path.is_absolute():
        # Resolve relative to repo root
        pdf_path = Path(__file__).resolve().parent.parent.parent / pdf_path
    if not pdf_path.exists():
        print(f"[eval] PDF not found: {pdf_path}", file=sys.stderr)
        return 1

    try:
        from buysheet_v2.pipeline import run_pipeline
    except ImportError:
        print(f"[eval] pipeline.py not yet implemented (Phase 2). Cannot run extraction.", file=sys.stderr)
        return 2

    result = run_pipeline(pdf_path, vendor_key=vendor)
    extracted_cards = [c.model_dump() for c in result.all_cards]
    report = compute_accuracy(golden, extracted_cards)
    print(format_report(report))
    # Return code reflects ship gate
    per_cell = report["per_cell_accuracy"] or 0
    per_card = report["per_card_accuracy"] or 0
    if per_cell >= 0.85 and per_card >= 0.90:
        return 0
    return 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <vendor>", file=sys.stderr)
        sys.exit(64)
    vendor = sys.argv[1]
    repo = Path(__file__).resolve().parent.parent.parent
    golden_path = repo / "buysheet_v2" / "tests" / "golden" / f"{vendor}.json"
    holdout_path = repo / "buysheet_v2" / "tests" / "holdout" / f"{vendor}.json"
    if golden_path.exists():
        sys.exit(run_eval(vendor, golden_path))
    if holdout_path.exists():
        sys.exit(run_eval(vendor, holdout_path))
    print(f"no golden/holdout for {vendor}", file=sys.stderr)
    sys.exit(1)
