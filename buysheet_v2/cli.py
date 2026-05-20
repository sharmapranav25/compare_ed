"""Command-line entry point for buysheet_v2.

Subcommands (Phase 0 only declares stubs; later phases implement them):

  buysheet_v2 run <pdf> [--vendor-key X]   # full pipeline -> BUYSHEET_<vendor>.xlsx
  buysheet_v2 debug-cards <pdf> --page N   # render page + overlay detected card bboxes
  buysheet_v2 eval <vendor>                # run pipeline + compare to tests/golden/<vendor>.json
  buysheet_v2 cost <pdf>                   # report per-phase token/dollar breakdown
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _cmd_run(args: argparse.Namespace) -> int:
    from buysheet_v2.pipeline import run_pipeline
    from buysheet_v2.write import write_workbook

    pdf_path = args.pdf.resolve()
    if not pdf_path.exists():
        print(f"[run] PDF not found: {pdf_path}", file=sys.stderr)
        return 1
    vendor_key = args.vendor_key or pdf_path.parent.name
    out_path = REPO / f"BUYSHEET_{vendor_key}_v2.xlsx"

    print(f"[run] vendor_key={vendor_key}  out={out_path.name}")
    extraction = run_pipeline(pdf_path, vendor_key=vendor_key)
    print(f"[run] writing workbook -> {out_path.name}")
    write_workbook(extraction, out_path, pdf_path=pdf_path)
    print(f"[run] DONE  -> {out_path}")
    print(f"      cards={len(extraction.all_cards)}  cost=${extraction.cost_usd:.3f}")
    return 0


def _cmd_debug_cards(args: argparse.Namespace) -> int:
    from buysheet_v2.cards import detect_cards_on_page, overlay_cards_on_page
    from buysheet_v2.ingest import ingest

    pdf_path = args.pdf.resolve()
    if not pdf_path.exists():
        print(f"[debug-cards] PDF not found: {pdf_path}", file=sys.stderr)
        return 1

    print(f"[debug-cards] ingesting {pdf_path.name}...", file=sys.stderr)
    doc = ingest(pdf_path)
    if args.page < 1 or args.page > doc.page_count:
        print(f"[debug-cards] page {args.page} out of range (1..{doc.page_count})", file=sys.stderr)
        return 1
    page = doc.pages[args.page - 1]
    print(f"[debug-cards] page {args.page}/{doc.page_count}  "
          f"({page.width_px}x{page.height_px}, {len(page.text)} chars text)", file=sys.stderr)

    cards, usage = detect_cards_on_page(page)
    print(f"[debug-cards] detected {len(cards)} cards  "
          f"(in={usage['input_tokens']} out={usage['output_tokens']} cache_read={usage['cache_read_tokens']})",
          file=sys.stderr)

    out_png_path = pdf_path.with_suffix("").with_name(
        f"{pdf_path.stem}.debug-cards-p{args.page:02d}.png"
    )
    overlay = overlay_cards_on_page(page, cards)
    out_png_path.write_bytes(overlay)
    print(f"[debug-cards] wrote {out_png_path}", file=sys.stderr)

    # Also print bbox details to stdout
    for i, c in enumerate(cards):
        print(f"  card #{i}  bbox={c.bbox_px}  sku_hint={c.sku_hint!r}")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from buysheet_v2.tests.eval_harness import run_eval

    golden_path = REPO / "buysheet_v2" / "tests" / "golden" / f"{args.vendor}.json"
    holdout_path = REPO / "buysheet_v2" / "tests" / "holdout" / f"{args.vendor}.json"
    if golden_path.exists():
        return run_eval(args.vendor, golden_path)
    if holdout_path.exists():
        return run_eval(args.vendor, holdout_path)
    print(f"[eval] no golden or holdout found for vendor: {args.vendor}", file=sys.stderr)
    print(f"       looked for: {golden_path}, {holdout_path}", file=sys.stderr)
    return 1


def _cmd_cost(args: argparse.Namespace) -> int:
    print(f"[cost] cost reporting not yet implemented (Phase 5) — would analyze: {args.pdf}", file=sys.stderr)
    return 2


def _cmd_backfill_photos(args: argparse.Namespace) -> int:
    """Re-run the YOLO+matcher on cached per-page sidecars, updating any card
    whose photo_bbox_px is unset. Useful after a fresh YOLO sidecar (new
    weights, re-detect) or after C1/C2's hi-res annotation lands — pick up
    new matches without re-extracting from scratch.

    Prereqs (on disk from a prior `buysheet_v2 run` invocation):
      <pdf>.v2.yolo.json      — YOLO bboxes + annotated paths
      <pdf>.v2.pages/NN.json  — per-page sidecars

    Outputs: per-page sidecars are updated in place with new photo_bbox_px
    values on cards previously missing one. Other fields are untouched.
    """
    from buysheet_v2.backfill import backfill_photos
    pdf_path: Path = args.pdf
    if not pdf_path.exists():
        print(f"[backfill] PDF not found: {pdf_path}", file=sys.stderr)
        return 1
    summary = backfill_photos(pdf_path, force=args.force, verbose=True)
    # Non-zero exit if nothing was scannable — useful for CI / scripting
    if summary["pages_scanned"] == 0:
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="buysheet_v2")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run full pipeline on a PDF")
    p_run.add_argument("pdf", type=Path)
    p_run.add_argument("--vendor-key", help="defaults to slugified PDF stem")
    p_run.set_defaults(func=_cmd_run)

    p_dbg = sub.add_parser("debug-cards", help="render page + overlay detected card bboxes")
    p_dbg.add_argument("pdf", type=Path)
    p_dbg.add_argument("--page", type=int, required=True)
    p_dbg.set_defaults(func=_cmd_debug_cards)

    p_eval = sub.add_parser("eval", help="run pipeline + compare to golden/holdout")
    p_eval.add_argument("vendor", help="vendor key (matches tests/{golden,holdout}/<vendor>.json)")
    p_eval.set_defaults(func=_cmd_eval)

    p_cost = sub.add_parser("cost", help="per-phase token/dollar breakdown")
    p_cost.add_argument("pdf", type=Path)
    p_cost.set_defaults(func=_cmd_cost)

    p_bf = sub.add_parser(
        "backfill-photos",
        help="re-run YOLO+matcher on cached per-page sidecars to fill missing photo_bbox_px",
    )
    p_bf.add_argument("pdf", type=Path,
                      help="the original PDF whose .v2.yolo.json + .v2.pages/ live alongside it")
    p_bf.add_argument(
        "--force", action="store_true",
        help="re-run matcher even on cards that already have photo_bbox_px set",
    )
    p_bf.set_defaults(func=_cmd_backfill_photos)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
