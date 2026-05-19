"""Single-page layout probe — the testing entry point for the
YOLO + PyMuPDF + matching flow.

Two modes:

  Mode 1 — PNG only (no PDF, no SKUs):
    python probe_layout.py path/to/page.png
    Runs YOLO + overlay. No SKU<->photo links. Useful for sanity-checking
    YOLO model loading, weights download, and the overlay renderer.

  Mode 2 — PDF + page (one-page version of the full flow):
    python probe_layout.py path/to/doc.pdf 7
    Renders the fullres PNG, runs YOLO + PyMuPDF observations, optionally
    runs the VLM extract + validator if anthropic credentials are
    available, and produces a per-page audit overlay + match.json.

Outputs land under probes/<doc-stem>/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import fitz
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _layout import (  # noqa: E402
    calibrate_per_vendor,
    calibrate_pymupdf_per_vendor,
    detect_yolo,
)
from _overlay import render_audit, render_index  # noqa: E402
from _pymupdf_obs import observations as pymupdf_observations  # noqa: E402
from _render import render_page_png_fullres  # noqa: E402

load_dotenv()

PROBES_DIR = Path(__file__).resolve().parent / "probes"


def _mode_png(png_path: Path, force: bool) -> None:
    """Mode 1: YOLO + overlay on a single PNG; no matching."""
    out_dir = PROBES_DIR / png_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    yolo_path = out_dir / "yolo.json"
    audit_path = out_dir / "audit.png"

    if yolo_path.exists() and not force:
        yolo = json.loads(yolo_path.read_text())
        print(f"using cached yolo.json ({len(yolo.get('boxes') or [])} boxes)",
              file=sys.stderr)
    else:
        boxes = detect_yolo(png_path)
        yolo = {"boxes": boxes}
        yolo_path.write_text(json.dumps(yolo, indent=2))
        print(f"yolo: {len(boxes)} boxes -> {yolo_path}", file=sys.stderr)

    # Stub calibration from this single page so the audit overlay can
    # show "kept" vs "dropped" markers.
    size_stats = calibrate_per_vendor(
        [{"page_no": 1, "boxes": yolo.get("boxes") or []}],
        [{"page_no": 1, "vendor": "__probe__", "label": "product"}],
    )
    stats = size_stats.get("__probe__") or size_stats.get("__doc_wide__") or {}
    # Build a fake match-log with just the filter decisions (no skus).
    from _layout import filter_candidates_by_size, figure_boxes  # local import
    y_filter = filter_candidates_by_size(
        figure_boxes(yolo.get("boxes") or []), stats,
    )
    match_log = {
        "filtered_photo_candidates": [
            {"source": "yolo", "bbox_px": f["box"]["bbox"], "kept": f["kept"],
             "reason": f.get("reason")} for f in y_filter
        ],
        "skus": [],
    }
    render_audit(png_path, yolo, None, match_log, audit_path)
    print(f"audit -> {audit_path}", file=sys.stderr)


def _mode_pdf(pdf_path: Path, page_no: int, force: bool) -> None:
    """Mode 2: full per-page flow on a single PDF page."""
    out_dir = PROBES_DIR / pdf_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    fullres_path = out_dir / f"{page_no:02d}.fullres.png"
    yolo_path = out_dir / f"{page_no:02d}.yolo.json"
    pymupdf_path = out_dir / f"{page_no:02d}.pymupdf.json"
    audit_path = out_dir / f"{page_no:02d}.audit.png"
    match_path = out_dir / f"{page_no:02d}.match.json"
    page_record_path = out_dir / f"{page_no:02d}.json"

    pdf = fitz.open(str(pdf_path))
    try:
        if not (1 <= page_no <= pdf.page_count):
            raise ValueError(f"page {page_no} out of range 1..{pdf.page_count}")
        if not fullres_path.exists() or force:
            render_page_png_fullres(pdf, page_no, fullres_path)
            print(f"fullres -> {fullres_path}", file=sys.stderr)

        if yolo_path.exists() and not force:
            yolo = json.loads(yolo_path.read_text())
        else:
            yolo = {"boxes": detect_yolo(fullres_path)}
            yolo_path.write_text(json.dumps(yolo, indent=2))
        print(f"yolo: {len(yolo.get('boxes') or [])} boxes", file=sys.stderr)

        if pymupdf_path.exists() and not force:
            pymupdf = json.loads(pymupdf_path.read_text())
        else:
            pymupdf = pymupdf_observations(pdf, page_no)
            pymupdf_path.write_text(json.dumps(pymupdf, indent=2))
        print(f"pymupdf: {len(pymupdf.get('words') or [])} words, "
              f"{len(pymupdf.get('image_rects') or [])} image rects",
              file=sys.stderr)

        # Try VLM extract — but it's optional in probe mode.
        page_record = _try_vlm_extract(pdf_path, page_no, fullres_path,
                                       force=force,
                                       cache_path=page_record_path)
    finally:
        pdf.close()

    # Build a minimal stub "records" list of one page so we can reuse the
    # full match_pass1 + calibrate + match_pass2 flow without modification.
    from match_photos import (  # noqa: PLC0415
        compute_recurring_xrefs,
        match_pass1, match_pass2, calibrate_distance_direction,
    )
    records = [{
        "page_no": page_no,
        "page": page_record,
        "page_path": page_record_path,
        "yolo": yolo,
        "pymupdf": pymupdf,
        "fullres": fullres_path,
    }]
    yolo_pages = [{"page_no": page_no, "boxes": yolo.get("boxes") or []}]
    pymupdf_pages_calib = [{"page_no": page_no,
                            "image_rects": pymupdf.get("image_rects") or []}]
    page_recs = [{"page_no": page_no,
                  "vendor": (page_record.get("prev_context") or {}).get("vendor") or "__probe__",
                  "label": "product"}]
    size_stats = calibrate_per_vendor(yolo_pages, page_recs)
    pymupdf_size_stats = calibrate_pymupdf_per_vendor(
        pymupdf_pages_calib, page_recs,
    )
    xref_pages = compute_recurring_xrefs(records)
    pairings = match_pass1(records, size_stats, pymupdf_size_stats,
                           xref_pages, 1)
    dist_stats = calibrate_distance_direction(pairings)
    match = match_pass2(records[0], size_stats, pymupdf_size_stats,
                        dist_stats, xref_pages, 1)
    match_path.write_text(json.dumps(match, indent=2))
    page_record_path.write_text(json.dumps(page_record, indent=2))
    render_audit(fullres_path, yolo, pymupdf, match, audit_path)

    matched = sum(1 for s in match["skus"]
                  if (s.get("match_combined") or {}).get("photo_bbox_px"))
    print(f"matched {matched}/{len(match['skus'])} skus -> {match_path}",
          file=sys.stderr)
    print(f"audit -> {audit_path}", file=sys.stderr)

    # Always (re)build the per-probe HTML index so the user can browse.
    index_path = out_dir / "_audit_index.html"
    calib_for_index = {match["vendor_used"]:
                       match["calibration_used"].get("pymupdf_size", {})}
    render_index(out_dir, index_path, calibration=calib_for_index)
    print(f"index -> {index_path}", file=sys.stderr)


def _try_vlm_extract(pdf_path: Path, page_no: int, fullres_path: Path,
                     force: bool, cache_path: Path) -> dict:
    """Best-effort VLM extract for probe Mode 2. Falls back to an empty
    page record if anthropic isn't configured or the call fails — the
    rest of the probe still runs."""
    if cache_path.exists() and not force:
        return json.loads(cache_path.read_text())
    record = {"page_no": page_no, "label": "product",
              "prev_context": {"vendor": "__probe__", "current_section": None},
              "context_after": {"vendor": "__probe__", "current_section": None},
              "products": []}
    try:
        import anthropic
        from extract_products import (  # local import to keep probe importable
            MODEL_OPUS, ensure_png, extract_page, validate_skus,
        )
        png_path = fullres_path.parent / f"{page_no:02d}.png"
        if not png_path.exists():
            pdf = fitz.open(str(pdf_path))
            try:
                ensure_png(pdf, page_no, png_path)
            finally:
                pdf.close()
        client = anthropic.Anthropic(max_retries=4)
        candidates = extract_page(client, MODEL_OPUS, png_path,
                                  record["prev_context"])
        products, rejected = validate_skus(client, candidates)
        record["products"] = products
        record["rejected_candidates"] = rejected
        print(f"vlm: {len(products)} products extracted "
              f"({len(rejected)} rejected)", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001
        print(f"vlm extract skipped ({type(exc).__name__}: {exc}) — "
              f"matching will run on 0 SKUs", file=sys.stderr)
    return record


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Probe the YOLO + PyMuPDF + matching layout pipeline."
    )
    ap.add_argument("input", type=Path,
                    help="PNG (Mode 1) or PDF (Mode 2)")
    ap.add_argument("page", type=int, nargs="?", default=None,
                    help="1-indexed page number (PDF mode only)")
    ap.add_argument("--force", action="store_true",
                    help="Re-run even if cached outputs exist")
    args = ap.parse_args()
    if not args.input.exists():
        ap.error(f"input not found: {args.input}")
    suffix = args.input.suffix.lower()
    if suffix == ".pdf":
        if args.page is None:
            ap.error("PDF mode requires a page number")
        _mode_pdf(args.input, args.page, args.force)
    elif suffix in (".png", ".jpg", ".jpeg"):
        _mode_png(args.input, args.force)
    else:
        ap.error(f"unsupported input extension: {suffix}")


if __name__ == "__main__":
    main()
