"""Multi-model extraction: produce 3 sidecars + 3 xlsx files per PDF.

For a fair head-to-head, all 3 models extract from the SAME page images and
get the SAME card-bbox hints. Only the per-page extract step differs.
Sonnet's cards.py outputs are reused for all three so card detection isn't
itself a variable in the comparison.

Workflow per PDF:
  1. Ingest PDF (page renders + text)
  2. classify.py (Opus, 1 call) — layout type, multi-brand?
  3. cards.py (Sonnet, N calls) — per-page bbox detection
  4. Extract with Sonnet (using existing extract.py) -> sidecar_sonnet
  5. Extract with Opus (same extract.py + model='claude-opus-4-7') -> sidecar_opus
  6. Extract with Gemini (extract_gemini.py) -> sidecar_gemini
  7. Write 3 xlsx files (one per sidecar)

Each xlsx is independently produced via the existing write.py pipeline so
buyer-facing artifacts mirror what each model would produce as the
"primary extractor". Compare via tools/three_way_compare.

Usage:
    python -m buysheet_v2.tools.multi_model_extract <pdf> --vendor-key <key>
    python -m buysheet_v2.tools.multi_model_extract <pdf> --skip sonnet   (reuse cached)
    python -m buysheet_v2.tools.multi_model_extract <pdf> --estimate     (dry-run cost)

Output layout (in same dir as PDF):
    <stem>.sonnet.v2.cards.json
    <stem>.opus.v2.cards.json
    <stem>.gemini.v2.cards.json
    BUYSHEET_<vendor>_v2_sonnet.xlsx
    BUYSHEET_<vendor>_v2_opus.xlsx
    BUYSHEET_<vendor>_v2_gemini.xlsx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import anthropic

from buysheet_v2.cards import detect_cards_on_page
from buysheet_v2.classify import classify_layout
from buysheet_v2.consistency import (
    deterministic_fill, detect_catalog_brand,
    drop_invalid_sku_cards, normalize_extraction,
)
from buysheet_v2.extract import extract_with_retry
from buysheet_v2.extract_gemini import (
    DEFAULT_MODEL as GEMINI_MODEL, extract_cards_on_page_gemini,
)
from buysheet_v2.ingest import ingest
from buysheet_v2.schemas.extraction_result import (
    CatalogExtraction, PageExtraction,
)
from buysheet_v2.verify import verify_catalog

# Per-MTok pricing for cost estimates
PRICING = {
    "sonnet": (3.0, 15.0, "claude-sonnet-4-6"),
    "opus":   (15.0, 75.0, "claude-opus-4-7"),
    "gemini": (0.30, 2.50, GEMINI_MODEL),
}


def estimate_cost_per_model(num_pages: int) -> dict:
    """Rough cost estimate per page. Heuristic from observed real runs."""
    # Per-page tokens (observed from today's runs):
    # input ~6000 (image + prompt + bbox payload), output ~2000 (structured JSON)
    in_per_page, out_per_page = 6000, 2000
    out = {}
    for label, (in_rate, out_rate, model_name) in PRICING.items():
        cost = num_pages * (in_per_page / 1e6 * in_rate + out_per_page / 1e6 * out_rate)
        out[label] = (cost, model_name)
    return out


def run_extraction_with_anthropic_model(
    doc, layout, model_label: str, model_name: str,
    client: anthropic.Anthropic, card_bboxes_by_page: dict,
    pdf_path: Path, vendor_key: str,
) -> CatalogExtraction:
    """Reuse cached card bboxes; re-run only the extract step with given model."""
    result = CatalogExtraction(pdf_path=str(pdf_path), vendor_key=vendor_key)
    result.layout = layout

    for page in doc.pages:
        bboxes = card_bboxes_by_page.get(page.page_no, [])
        try:
            cards, usage = extract_with_retry(
                page, bboxes, client=client,
                layout_type=layout.layout_type,
                is_multi_brand=layout.is_multi_brand,
                expected_fields=layout.expected_fields_present,
                catalog_brand=(
                    None if layout.is_multi_brand
                    else (layout.notes or "").split(":")[0].strip() or None
                ),
                model=model_name,
            )
            result.pages.append(PageExtraction(
                page=page.page_no, cards=cards, page_text=page.text,
            ))
            result.tokens_input += usage["input_tokens"]
            result.tokens_output += usage["output_tokens"]
            result.tokens_cache_read += usage["cache_read_tokens"]
            in_rate, out_rate, _ = PRICING[model_label]
            result.cost_usd += (
                (usage["input_tokens"] / 1e6) * in_rate
                + (usage["output_tokens"] / 1e6) * out_rate
                + (usage["cache_read_tokens"] / 1e6) * in_rate * 0.1
            )
            print(f"  [{model_label}] page {page.page_no}/{doc.page_count}  "
                  f"cards={len(cards)}  cumul ${result.cost_usd:.3f}")
        except Exception as e:
            print(f"  [{model_label}] page {page.page_no} FAILED ({type(e).__name__}: {e})")
            result.pages.append(PageExtraction(page=page.page_no, error=str(e)))

    return result


def run_extraction_with_gemini(
    doc, layout, card_bboxes_by_page: dict,
    pdf_path: Path, vendor_key: str,
) -> CatalogExtraction:
    """Same as above but with Gemini. card_bboxes not passed (Gemini doesn't use them)."""
    result = CatalogExtraction(pdf_path=str(pdf_path), vendor_key=vendor_key)
    result.layout = layout

    for page in doc.pages:
        bboxes = card_bboxes_by_page.get(page.page_no, [])
        if not bboxes:
            result.pages.append(PageExtraction(page=page.page_no, page_text=page.text))
            continue
        try:
            cards, usage = extract_cards_on_page_gemini(
                page, bboxes,
                layout_type=layout.layout_type,
                is_multi_brand=layout.is_multi_brand,
                expected_fields=layout.expected_fields_present,
                catalog_brand=(
                    None if layout.is_multi_brand
                    else (layout.notes or "").split(":")[0].strip() or None
                ),
            )
            result.pages.append(PageExtraction(
                page=page.page_no, cards=cards, page_text=page.text,
            ))
            result.tokens_input += usage["input_tokens"]
            result.tokens_output += usage["output_tokens"]
            result.cost_usd += usage["cost_usd"]
            print(f"  [gemini] page {page.page_no}/{doc.page_count}  "
                  f"cards={len(cards)}  cumul ${result.cost_usd:.3f}")
        except Exception as e:
            print(f"  [gemini] page {page.page_no} FAILED ({type(e).__name__}: {e})")
            result.pages.append(PageExtraction(page=page.page_no, error=str(e)))

    return result


def finalize_and_write(
    result: CatalogExtraction, sidecar_path: Path, xlsx_path: Path,
    pdf_path: Path,
) -> None:
    """Apply post-processing + write sidecar + xlsx."""
    # Drop fake SKUs (matches main pipeline behavior)
    for pe in result.pages:
        survivors, _ = drop_invalid_sku_cards(pe.cards)
        pe.cards = survivors
    normalize_extraction(result.all_cards)
    page_text_by = {p.page: p.page_text or "" for p in result.pages}
    deterministic_fill(result.all_cards, page_text_by,
                       catalog_brand=detect_catalog_brand(result.all_cards))
    result = verify_catalog(result)

    sidecar_path.write_text(result.model_dump_json(indent=2, exclude_none=False))
    print(f"  wrote sidecar: {sidecar_path.name} ({len(result.all_cards)} cards)")

    # Late import so unrelated tools don't pay the openpyxl cost
    from buysheet_v2.write import write_workbook
    write_workbook(result, xlsx_path, pdf_path=pdf_path, embed_photos=False)
    print(f"  wrote xlsx:    {xlsx_path.name}")


def main() -> int:
    ap = argparse.ArgumentParser(prog="multi_model_extract")
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--vendor-key", required=True)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Where to write sidecars + xlsx (default: same dir as PDF)")
    ap.add_argument("--skip", choices=["sonnet", "opus", "gemini"], action="append",
                    default=[], help="Skip a specific model (e.g. if you already have it)")
    ap.add_argument("--estimate", action="store_true",
                    help="Print cost estimate and exit")
    args = ap.parse_args()

    if not args.pdf.exists():
        print(f"PDF not found: {args.pdf}", file=sys.stderr)
        return 1

    out_dir = args.out_dir or args.pdf.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.pdf.stem

    print(f"\n[multi_model] {args.pdf.name}")
    print(f"  out dir: {out_dir}")

    if args.estimate:
        # Just count pages first
        doc = ingest(args.pdf)
        ests = estimate_cost_per_model(doc.page_count)
        print(f"  pages: {doc.page_count}")
        total = 0.0
        for label, (cost, model_name) in ests.items():
            if label in args.skip:
                print(f"  {label:8} ({model_name:25}) SKIPPED")
                continue
            print(f"  {label:8} ({model_name:25}) ~${cost:6.2f}")
            total += cost
        print(f"  TOTAL ESTIMATE: ~${total:.2f}")
        return 0

    client = anthropic.Anthropic()

    print(f"  ingesting + classify + cards (shared across all models)...")
    doc = ingest(args.pdf)
    layout, _ = classify_layout(doc, client=client)
    print(f"  layout={layout.layout_type}  multi_brand={layout.is_multi_brand}")

    card_bboxes_by_page = {}
    for page in doc.pages:
        try:
            bboxes, _ = detect_cards_on_page(
                page, client=client,
                layout_type=layout.layout_type,
                is_multi_brand=layout.is_multi_brand,
            )
            card_bboxes_by_page[page.page_no] = bboxes
        except Exception as e:
            print(f"  cards.py FAILED on page {page.page_no}: {e}")
            card_bboxes_by_page[page.page_no] = []
    total_bboxes = sum(len(v) for v in card_bboxes_by_page.values())
    print(f"  card_bboxes detected: {total_bboxes} across {doc.page_count} pages")

    if "sonnet" not in args.skip:
        print(f"\n[sonnet] running...")
        sonnet_ext = run_extraction_with_anthropic_model(
            doc, layout, "sonnet", "claude-sonnet-4-6",
            client, card_bboxes_by_page, args.pdf, args.vendor_key,
        )
        finalize_and_write(
            sonnet_ext,
            out_dir / f"{stem}.sonnet.v2.cards.json",
            out_dir / f"BUYSHEET_{args.vendor_key}_sonnet.xlsx",
            args.pdf,
        )

    if "opus" not in args.skip:
        print(f"\n[opus] running...")
        opus_ext = run_extraction_with_anthropic_model(
            doc, layout, "opus", "claude-opus-4-7",
            client, card_bboxes_by_page, args.pdf, args.vendor_key,
        )
        finalize_and_write(
            opus_ext,
            out_dir / f"{stem}.opus.v2.cards.json",
            out_dir / f"BUYSHEET_{args.vendor_key}_opus.xlsx",
            args.pdf,
        )

    if "gemini" not in args.skip:
        print(f"\n[gemini] running...")
        gemini_ext = run_extraction_with_gemini(
            doc, layout, card_bboxes_by_page, args.pdf, args.vendor_key,
        )
        finalize_and_write(
            gemini_ext,
            out_dir / f"{stem}.gemini.v2.cards.json",
            out_dir / f"BUYSHEET_{args.vendor_key}_gemini.xlsx",
            args.pdf,
        )

    print(f"\n[multi_model] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
