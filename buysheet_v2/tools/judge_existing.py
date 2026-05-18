"""Run VLM-as-judge against an existing extraction sidecar (no Sonnet re-run).

For ad-hoc validation: given a `.v2.cards.json` and the source PDF that
produced it, run Opus 4.7 over each page, compare per-card per-field with
the cached Sonnet extraction, and emit a structured agreement report.

Useful for:
  - Validating accuracy on a cached run without re-paying for Sonnet
  - Comparing agreement rates across multiple vendors
  - Bootstrapping a "double-verified" subset of cells for ground truth

Usage:
    # Judge a single sidecar (PDF inferred from "pdf_path" stored in sidecar)
    python -m buysheet_v2.tools.judge_existing files/<vendor>/<doc>.v2.cards.json

    # Judge multiple sidecars, write enhanced versions back in place
    python -m buysheet_v2.tools.judge_existing --update files/<vendor>/*.v2.cards.json

    # Just the cost estimate, no API calls
    python -m buysheet_v2.tools.judge_existing --estimate <sidecar>.v2.cards.json
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import anthropic

from buysheet_v2.ingest import ingest
from buysheet_v2.judge import (
    OPUS_IN_PER_MTOK,
    OPUS_OUT_PER_MTOK,
    judge_page,
    merge_judge_into_confidence,
)
from buysheet_v2.schemas.extraction_result import (
    CardConfidence, CatalogExtraction,
)


def estimate_cost(num_pages: int) -> float:
    """Rough cost estimate. Opus per page ~ 5000 in + 2000 out tokens."""
    in_cost = (num_pages * 5000 / 1e6) * OPUS_IN_PER_MTOK
    out_cost = (num_pages * 2000 / 1e6) * OPUS_OUT_PER_MTOK
    return in_cost + out_cost


def judge_one(sidecar: Path, *, update_in_place: bool = False) -> dict:
    """Judge one sidecar. Returns agreement summary."""
    ext = CatalogExtraction.model_validate_json(sidecar.read_text())
    pdf_path = Path(ext.pdf_path)
    if not pdf_path.exists():
        # Fall back to PDF in the same directory as the sidecar
        candidates = list(sidecar.parent.glob("*.pdf"))
        if not candidates:
            raise FileNotFoundError(
                f"PDF not found at {ext.pdf_path} or in {sidecar.parent}",
            )
        pdf_path = candidates[0]

    print(f"\n[judge] {sidecar.name}")
    print(f"  PDF: {pdf_path.name}  pages: {len(ext.pages)}  cards: {len(ext.all_cards)}")

    print(f"  Ingesting page images (PyMuPDF render)...")
    doc = ingest(pdf_path)
    client = anthropic.Anthropic()

    judge_by_page = {}
    total_cost = 0.0
    for i, page in enumerate(doc.pages, 1):
        page_extract = next((pe for pe in ext.pages if pe.page == page.page_no), None)
        if not page_extract or not page_extract.cards:
            continue
        print(f"  page {page.page_no}/{len(doc.pages)}  judging {len(page_extract.cards)} cards...",
              end=" ", flush=True)
        try:
            jcards, jusage = judge_page(
                page, page_extract.cards, card_bboxes=[], client=client,
            )
            judge_by_page[page.page_no] = jcards
            total_cost += jusage.get("cost_usd", 0.0)
            print(f"-> {len(jcards)} judge cards  ${jusage['cost_usd']:.3f}")
        except Exception as e:
            print(f"FAILED ({type(e).__name__}: {e})")

    # Initialize confidence list if missing
    if not ext.confidence:
        ext.confidence = [
            CardConfidence(sku=c.sku, page=c.page) for c in ext.all_cards
        ]
    counts = merge_judge_into_confidence(
        ext.confidence, ext.all_cards, judge_by_page,
    )

    total = counts["total_compared"] or 1
    print(f"\n  === JUDGE RESULT ===")
    print(f"  Total field comparisons: {counts['total_compared']}")
    print(f"  Agreed (Sonnet == Opus): {counts['agree']:>6} ({100*counts['agree']/total:.1f}%)")
    print(f"  Disagreed:                {counts['disagree']:>6} ({100*counts['disagree']/total:.1f}%)")
    print(f"  Asymmetric (one has, other doesn't): {counts['asymmetric']:>3}")
    print(f"  Total Opus cost: ${total_cost:.3f}")

    # Per-field agreement breakdown
    print(f"\n  Per-field agreement:")
    per_field_counts: dict[str, Counter] = {}
    for conf in ext.confidence:
        for field, score in (conf.judge_agreement or {}).items():
            per_field_counts.setdefault(field, Counter())
            if score >= 1.0:
                per_field_counts[field]["agree"] += 1
            elif score <= 0.0:
                per_field_counts[field]["asymmetric"] += 1
            else:
                per_field_counts[field]["disagree"] += 1
    for field in sorted(per_field_counts):
        c = per_field_counts[field]
        ftot = sum(c.values())
        if ftot == 0:
            continue
        print(f"    {field:<14}  agree={c['agree']}/{ftot} = {100*c['agree']/ftot:5.1f}%   "
              f"disagree={c['disagree']}  asymmetric={c['asymmetric']}")

    # Show sample disagreements
    print(f"\n  Sample disagreements (Sonnet -> Opus):")
    samples_shown = 0
    card_by_key = {(c.sku, c.page): c for c in ext.all_cards}
    for conf in ext.confidence:
        if samples_shown >= 10:
            break
        card = card_by_key.get((conf.sku, conf.page))
        for field, score in (conf.judge_agreement or {}).items():
            if 0.0 < score < 1.0:
                s_val = getattr(card, field, None)
                j_val = conf.judge_values.get(field)
                print(f"    {conf.sku:<14} p{conf.page:<3} {field:<14} "
                      f"Sonnet={s_val!r:35} Opus={j_val!r}")
                samples_shown += 1
                if samples_shown >= 10:
                    break

    if update_in_place:
        sidecar.write_text(ext.model_dump_json(indent=2, exclude_none=False))
        print(f"\n  -> updated {sidecar.name} in place with judge_agreement data")

    return {
        "sidecar": sidecar.name,
        "total_comparisons": counts["total_compared"],
        "agree": counts["agree"],
        "disagree": counts["disagree"],
        "asymmetric": counts["asymmetric"],
        "cost_usd": total_cost,
    }


def main() -> int:
    ap = argparse.ArgumentParser(prog="judge_existing")
    ap.add_argument("sidecars", nargs="+", type=Path)
    ap.add_argument("--update", action="store_true",
                    help="Write judge_agreement back into the sidecar JSON")
    ap.add_argument("--estimate", action="store_true",
                    help="Estimate cost without running")
    args = ap.parse_args()

    results = []
    for sc in args.sidecars:
        if not sc.exists():
            print(f"[skip] {sc} (not found)", file=sys.stderr)
            continue
        if args.estimate:
            try:
                ext = CatalogExtraction.model_validate_json(sc.read_text())
                cost = estimate_cost(len(ext.pages))
                print(f"{sc.name}: {len(ext.pages)} pages, est cost ~${cost:.2f}")
            except Exception as e:
                print(f"{sc.name}: estimate failed ({type(e).__name__}: {e})")
            continue
        results.append(judge_one(sc, update_in_place=args.update))

    if results:
        total_cost = sum(r["cost_usd"] for r in results)
        total_compared = sum(r["total_comparisons"] for r in results)
        total_agree = sum(r["agree"] for r in results)
        print(f"\n=== AGGREGATE across {len(results)} sidecar(s) ===")
        print(f"  Total comparisons: {total_compared}")
        print(f"  Total agreement:   {total_agree}/{total_compared} = "
              f"{100*total_agree/max(1,total_compared):.1f}%")
        print(f"  Total Opus cost:   ${total_cost:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
