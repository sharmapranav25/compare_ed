"""Scaffold a ground-truth golden test set from a current v2 extraction sidecar.

Replaces the older tests/scaffold_golden.py which read from v1 xlsx output
(stub-quality extractions). This version reads `.v2.cards.json` directly so
the scaffolded values are the BEST current extraction — the human reviewer
only has to confirm or correct, not start from scratch.

Output JSON shape — same as the existing eval harness expects:

    {
      "vendor_key": "...",
      "vendor_type": "athletic-grid|multi-brand|lookbook|image-only-ppt",
      "pdf_path": "...",
      "source_sidecar": "...",
      "description": "Scaffold from <sidecar> on <date>; flip `_verified` to true after verifying each SKU",
      "skus": [
        {
          "_verified": false,
          "_page": <int>,
          "_notes": null,
          "sku": "...",
          "brand": "...",
          "description": "...",
          ...
        },
        ...
      ]
    }

Usage:
    python -m buysheet_v2.tools.scaffold_golden_v2 \\
        ~/buysheet_runs/<vendor>/<timestamp>/<doc>.v2.cards.json \\
        --vendor-key nike_ho26 \\
        --vendor-type athletic-grid \\
        --sample 25

If a golden file already exists at tests/golden/<vendor>.json, the tool
refuses to overwrite unless --force is passed. This protects already-
verified entries from accidental loss.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import date
from pathlib import Path

from buysheet_v2.consistency import (
    deterministic_fill, detect_catalog_brand,
    drop_invalid_sku_cards, normalize_extraction,
)
from buysheet_v2.schemas.extraction_result import CatalogExtraction
from buysheet_v2.verify import verify_catalog

REPO = Path(__file__).resolve().parent.parent.parent
GOLDEN_DIR = REPO / "buysheet_v2" / "tests" / "golden"

FIELDS_TO_VERIFY = [
    "sku", "brand", "description", "color", "standard_color",
    "mg", "sg", "ssg", "intro_date", "usd_cost", "usd_retail",
]


def sample_evenly(cards: list, n: int = 25) -> list:
    """Pick n cards evenly distributed across the catalog (not just first/last)."""
    if len(cards) <= n:
        return list(cards)
    indices = [round(i * (len(cards) - 1) / (n - 1)) for i in range(n)]
    seen = set()
    deduped = []
    for i in indices:
        if i not in seen:
            seen.add(i)
            deduped.append(cards[i])
    return deduped


def main() -> int:
    ap = argparse.ArgumentParser(prog="scaffold_golden_v2")
    ap.add_argument("sidecar", type=Path, help="Path to <doc>.v2.cards.json")
    ap.add_argument("--vendor-key", required=True,
                    help="Short key like nike_ho26 (becomes the JSON filename)")
    ap.add_argument("--vendor-type", required=True,
                    choices=["athletic-grid", "multi-brand", "lookbook",
                             "image-only-ppt", "apparel-marketing", "other"],
                    help="Catalog layout family (for evaluation grouping)")
    ap.add_argument("--sample", type=int, default=25,
                    help="Number of SKUs to sample (default 25)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for sample stability")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing golden file even if it has verified entries")
    args = ap.parse_args()

    if not args.sidecar.exists():
        print(f"sidecar not found: {args.sidecar}", file=sys.stderr)
        return 1

    out_path = GOLDEN_DIR / f"{args.vendor_key}.json"
    if out_path.exists() and not args.force:
        # Check if existing has verified entries — refuse to overwrite if so
        existing = json.loads(out_path.read_text())
        existing_verified = sum(1 for s in existing.get("skus", []) if s.get("_verified"))
        if existing_verified:
            print(f"ERROR: {out_path} has {existing_verified} verified entries; "
                  f"refusing to overwrite. Use --force if you really mean it.",
                  file=sys.stderr)
            return 2

    # Load + post-process the sidecar so we sample from the CURRENT-BEST extraction
    ext = CatalogExtraction.model_validate_json(args.sidecar.read_text())
    for pe in ext.pages:
        survivors, _ = drop_invalid_sku_cards(pe.cards)
        pe.cards = survivors
    normalize_extraction(ext.all_cards)
    page_text_by = {p.page: p.page_text or "" for p in ext.pages}
    deterministic_fill(ext.all_cards, page_text_by,
                       catalog_brand=detect_catalog_brand(ext.all_cards))
    ext = verify_catalog(ext)

    random.seed(args.seed)
    cards = list(ext.all_cards)
    sampled = sample_evenly(cards, args.sample)

    golden_skus = []
    for c in sampled:
        entry = {
            "_verified": False,
            "_page": c.page,
            "_notes": None,
        }
        for f in FIELDS_TO_VERIFY:
            entry[f] = getattr(c, f, None)
        golden_skus.append(entry)

    golden = {
        "vendor_key": args.vendor_key,
        "vendor_type": args.vendor_type,
        "pdf_path": ext.pdf_path,
        "source_sidecar": str(args.sidecar),
        "scaffolded_at": date.today().isoformat(),
        "description": (
            f"Scaffolded from {args.sidecar.name} ({args.sample} SKUs sampled "
            f"evenly across {len(ext.all_cards)} cards). Each entry has "
            f"`_verified: false` — flip to true after manually verifying field "
            f"values against the source PDF. Eval harness only counts verified "
            f"entries."
        ),
        "skus": golden_skus,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(golden, indent=2, default=str))
    print(f"wrote {out_path}")
    print(f"  vendor: {args.vendor_key} ({args.vendor_type})")
    print(f"  SKUs scaffolded: {len(golden_skus)} (of {len(ext.all_cards)} extracted)")
    print(f"  All entries `_verified: false` — run verify_golden tool to mark verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
