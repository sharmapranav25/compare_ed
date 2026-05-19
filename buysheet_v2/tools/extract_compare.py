"""Head-to-head extraction comparison: Sonnet 4.6 vs Gemini 2.5 Pro.

Given an existing `.v2.cards.json` sidecar (from a Sonnet run) and the source
PDF that produced it, run Gemini extraction over the same pages and report:

  - per-vendor agreement rate (per-card per-field)
  - cost comparison (Sonnet vs Gemini)
  - per-field win/loss (which model extracted the verifiable value more often)
  - sample disagreements with both candidates side-by-side

Cost: depends on page count. Gemini 2.5 Pro is ~$0.025-0.05 per page (input
heavy due to image tokens; output is small structured JSON). Cheaper than
Sonnet's ~$0.05/page.

Usage:
    python -m buysheet_v2.tools.extract_compare path/to/<doc>.v2.cards.json
    python -m buysheet_v2.tools.extract_compare --model gemini-2.5-flash <sidecar>
    python -m buysheet_v2.tools.extract_compare --estimate <sidecar>
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

from buysheet_v2.extract_gemini import (
    DEFAULT_MODEL, GEMINI_PRICING, extract_cards_on_page_gemini,
)
from buysheet_v2.ingest import ingest
from buysheet_v2.schemas.card import ProductCard
from buysheet_v2.schemas.extraction_result import CatalogExtraction


# Fields where exact (normalized) string equality is meaningful for comparison.
COMPARE_FIELDS = ("sku", "brand", "description", "color", "mg", "sg", "ssg",
                  "standard_color", "intro_date", "usd_cost")


def _normalize(value) -> str:
    if value is None:
        return ""
    s = str(value).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def estimate_cost(num_pages: int, model: str = DEFAULT_MODEL) -> float:
    """Heuristic estimate. Gemini's image-token cost is the dominant input."""
    in_rate, out_rate = GEMINI_PRICING.get(model, (1.25, 5.00))
    # Per-page tokens: image is ~2000-4000 input tokens, prompt is ~1000,
    # output is ~500-2000 for structured JSON
    per_page_in = 4000
    per_page_out = 1200
    return num_pages * (per_page_in / 1e6 * in_rate + per_page_out / 1e6 * out_rate)


def index_cards(cards: list[ProductCard]) -> dict[tuple[int, str], ProductCard]:
    """Index cards by (page, normalized SKU) for cross-model alignment."""
    return {(c.page, _normalize(c.sku)): c for c in cards}


def compare_one(sidecar: Path, *, model: str = DEFAULT_MODEL) -> dict:
    ext = CatalogExtraction.model_validate_json(sidecar.read_text())
    pdf_path = Path(ext.pdf_path)
    if not pdf_path.exists():
        candidates = list(sidecar.parent.glob("*.pdf"))
        if not candidates:
            raise FileNotFoundError(f"PDF missing: {ext.pdf_path}")
        pdf_path = candidates[0]

    print(f"\n[compare] {sidecar.name}")
    print(f"  PDF: {pdf_path.name}  pages: {len(ext.pages)}")
    print(f"  Sonnet baseline: {len(ext.all_cards)} cards extracted, "
          f"${ext.cost_usd:.2f} Sonnet cost")
    print(f"  Running Gemini ({model})...")

    doc = ingest(pdf_path)
    gemini_cards: list[ProductCard] = []
    gemini_cost = 0.0
    pages_run = 0
    for page in doc.pages:
        page_extract = next((pe for pe in ext.pages if pe.page == page.page_no), None)
        if not page_extract or not page_extract.cards:
            # Sonnet found nothing here; skip — Gemini probably nothing too
            continue
        print(f"  page {page.page_no}/{len(doc.pages)}  "
              f"Sonnet had {len(page_extract.cards)} cards...",
              end=" ", flush=True)
        try:
            gcards, gusage = extract_cards_on_page_gemini(
                page, card_bboxes=[], model=model,
            )
            gemini_cards.extend(gcards)
            gemini_cost += gusage["cost_usd"]
            pages_run += 1
            print(f"-> Gemini found {len(gcards)} cards  ${gusage['cost_usd']:.3f}")
        except Exception as e:
            print(f"FAILED ({type(e).__name__}: {e})")

    sonnet_idx = index_cards(ext.all_cards)
    gemini_idx = index_cards(gemini_cards)

    sonnet_keys = set(sonnet_idx.keys())
    gemini_keys = set(gemini_idx.keys())
    both = sonnet_keys & gemini_keys
    sonnet_only = sonnet_keys - gemini_keys
    gemini_only = gemini_keys - sonnet_keys

    print(f"\n  === EXTRACTION COMPARISON ===")
    print(f"  Sonnet cards: {len(sonnet_keys)}  ·  Gemini cards: {len(gemini_keys)}")
    print(f"  Both models extracted same SKU: {len(both)}")
    print(f"  Sonnet-only SKUs (Gemini missed): {len(sonnet_only)}")
    print(f"  Gemini-only SKUs (Sonnet missed): {len(gemini_only)}")

    # Per-field agreement on cards both models extracted
    field_agreement: dict[str, Counter] = {f: Counter() for f in COMPARE_FIELDS}
    field_disagreements: dict[str, list] = {f: [] for f in COMPARE_FIELDS}
    for key in both:
        s = sonnet_idx[key]
        g = gemini_idx[key]
        for f in COMPARE_FIELDS:
            s_val = getattr(s, f, None)
            g_val = getattr(g, f, None)
            if s_val is None and g_val is None:
                field_agreement[f]["both_null"] += 1
            elif s_val is None or g_val is None:
                field_agreement[f]["one_null"] += 1
                if len(field_disagreements[f]) < 6:
                    field_disagreements[f].append((s.sku, s_val, g_val))
            elif _normalize(s_val) == _normalize(g_val):
                field_agreement[f]["agree"] += 1
            else:
                field_agreement[f]["disagree"] += 1
                if len(field_disagreements[f]) < 6:
                    field_disagreements[f].append((s.sku, s_val, g_val))

    print(f"\n  Per-field agreement (both models extracted this SKU):")
    print(f"  {'field':<16}  {'agree':>6}  {'disagree':>9}  {'one_null':>9}  {'both_null':>10}  {'agree %':>8}")
    for f in COMPARE_FIELDS:
        c = field_agreement[f]
        compared = c["agree"] + c["disagree"]
        agree_pct = 100 * c["agree"] / compared if compared else 0
        print(f"  {f:<16}  {c['agree']:>6}  {c['disagree']:>9}  "
              f"{c['one_null']:>9}  {c['both_null']:>10}  {agree_pct:>7.1f}%")

    # Cost comparison
    sonnet_per_page = ext.cost_usd / max(1, len(ext.pages))
    gemini_per_page = gemini_cost / max(1, pages_run)
    print(f"\n  === COST COMPARISON ===")
    print(f"  Sonnet (full extraction): ${ext.cost_usd:.2f} = ${sonnet_per_page:.3f}/page")
    print(f"  Gemini ({model}):         ${gemini_cost:.2f} = ${gemini_per_page:.3f}/page")
    if sonnet_per_page > 0:
        ratio = gemini_per_page / sonnet_per_page
        print(f"  Gemini is {ratio:.2f}x Sonnet cost per page "
              f"({100*(1-ratio):.1f}% cheaper)" if ratio < 1 else
              f"  Gemini is {ratio:.2f}x Sonnet cost per page")

    # Sample disagreements (description + color are most informative)
    for f in ("description", "color", "intro_date", "usd_cost"):
        if not field_disagreements[f]:
            continue
        print(f"\n  Sample {f} disagreements (Sonnet vs Gemini):")
        for sku, sv, gv in field_disagreements[f][:5]:
            print(f"    {sku:<18}  Sonnet={sv!r:35}  Gemini={gv!r}")

    return {
        "sidecar": sidecar.name,
        "sonnet_cards": len(sonnet_keys),
        "gemini_cards": len(gemini_keys),
        "both_extracted": len(both),
        "sonnet_only": len(sonnet_only),
        "gemini_only": len(gemini_only),
        "sonnet_cost": ext.cost_usd,
        "gemini_cost": gemini_cost,
        "field_agreement": {f: dict(c) for f, c in field_agreement.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser(prog="extract_compare")
    ap.add_argument("sidecars", nargs="+", type=Path)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"Gemini model (default: {DEFAULT_MODEL})")
    ap.add_argument("--estimate", action="store_true",
                    help="Estimate cost without running")
    args = ap.parse_args()

    results = []
    for sc in args.sidecars:
        if not sc.exists():
            print(f"[skip] {sc}", file=sys.stderr)
            continue
        if args.estimate:
            ext = CatalogExtraction.model_validate_json(sc.read_text())
            cost = estimate_cost(len(ext.pages), args.model)
            print(f"{sc.name}: {len(ext.pages)} pages, est cost ~${cost:.2f}")
            continue
        results.append(compare_one(sc, model=args.model))

    if results:
        total_gemini = sum(r["gemini_cost"] for r in results)
        total_sonnet = sum(r["sonnet_cost"] for r in results)
        print(f"\n=== AGGREGATE across {len(results)} catalog(s) ===")
        print(f"  Sonnet total: ${total_sonnet:.2f}")
        print(f"  Gemini total: ${total_gemini:.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
