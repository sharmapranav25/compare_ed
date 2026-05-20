"""Backfill photo_bbox_px on cached per-page sidecars.

When the YOLO + matcher pre-pass is re-run (e.g. after C1+C2's hi-res
annotation lands, or after a fresh YOLO weights change), the existing
per-page sidecars may carry cards with `photo_bbox_px=None` even though
YOLO would now detect a box for them. Rather than re-run the full
extract (Sonnet calls + verify + judge), backfill just walks the YOLO
sidecar + per-page sidecars and runs the matcher on the residual set.

Ports parent repo's extract_products._backfill_image_paths:702-744. Same
semantics — only touches cards whose products list exists AND no card
already has an `photo_bbox_px` set (leaves prior runs alone).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import anthropic

from buysheet_v2.photo_match import match_skus_to_bboxes
from buysheet_v2.sidecar import (
    list_cached_pages, load_page_sidecar, save_page_sidecar,
)
from buysheet_v2.sku_text_check import present_skus_in_text


def _load_yolo_sidecar(pdf_path: Path) -> dict[int, dict]:
    """Parse the YOLO sidecar into {page_no: detection_entry}."""
    sidecar_path = pdf_path.with_suffix("").with_name(
        f"{pdf_path.stem}.v2.yolo.json",
    )
    if not sidecar_path.exists():
        return {}
    try:
        raw = json.loads(sidecar_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    detections = raw.get("detections", {})
    out: dict[int, dict] = {}
    for k, v in detections.items():
        try:
            out[int(k)] = v
        except (TypeError, ValueError):
            continue
    return out


def backfill_photos(
    pdf_path: Path,
    *,
    client: Optional[anthropic.Anthropic] = None,
    force: bool = False,
    verbose: bool = True,
) -> dict[str, int]:
    """Re-run the matcher on cached per-page sidecars where cards lack
    photo_bbox_px.

    Inputs (on disk, all must pre-exist from a prior `run` invocation):
      <pdf_stem>.v2.yolo.json          — YOLO bboxes + annotated_paths
      <pdf_stem>.v2.pages/NN.json      — per-page PageExtractions

    Effect: updates each page sidecar in place, setting `photo_bbox_px`
    on cards that the matcher now assigns to a YOLO box. Other fields are
    untouched.

    `force=True` re-runs the matcher even on pages where ALL cards
    already have a `photo_bbox_px`. By default such pages are skipped.

    Returns a summary dict with `pages_scanned`, `pages_updated`,
    `cards_matched`, `cards_unchanged`.
    """
    yolo = _load_yolo_sidecar(pdf_path)
    cached_pages = list_cached_pages(pdf_path)
    summary = {
        "pages_scanned": 0, "pages_updated": 0,
        "cards_matched": 0, "cards_unchanged": 0,
    }
    if not yolo:
        if verbose:
            print(f"[backfill] no YOLO sidecar at "
                  f"{pdf_path.stem}.v2.yolo.json — nothing to do",
                  file=sys.stderr)
        return summary
    if not cached_pages:
        if verbose:
            print(f"[backfill] no per-page sidecars under "
                  f"{pdf_path.stem}.v2.pages/ — nothing to do",
                  file=sys.stderr)
        return summary
    if client is None:
        client = anthropic.Anthropic()

    for page_no in cached_pages:
        summary["pages_scanned"] += 1
        page_extraction = load_page_sidecar(pdf_path, page_no)
        if page_extraction is None or not page_extraction.cards:
            continue
        yolo_entry = yolo.get(page_no) or {}
        bboxes = yolo_entry.get("bboxes") or []
        annotated_path = yolo_entry.get("annotated_path")
        if not bboxes or not annotated_path:
            continue
        if not Path(annotated_path).exists():
            if verbose:
                print(f"[backfill] page {page_no}: annotated image missing "
                      f"({annotated_path}) — skipping", file=sys.stderr)
            continue
        # Skip if every card already has a bbox (unless force=True)
        if not force and all(c.photo_bbox_px for c in page_extraction.cards):
            continue

        # Apply same text-layer SKU filter the live pipeline uses
        present, text_layer_present = present_skus_in_text(
            page_extraction.page_text or "",
            [c.sku for c in page_extraction.cards],
        )
        if text_layer_present:
            matcher_input = [c for c in page_extraction.cards
                             if c.sku in present]
        else:
            matcher_input = list(page_extraction.cards)
        if not matcher_input:
            if verbose:
                print(f"[backfill] page {page_no}: no matcher input after "
                      f"text-layer filter — skipping", file=sys.stderr)
            continue

        try:
            matched = match_skus_to_bboxes(
                Path(annotated_path), matcher_input, bboxes, client=client,
            )
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"[backfill] page {page_no}: matcher FAILED "
                      f"({type(exc).__name__}: {exc})", file=sys.stderr)
            continue

        page_matched = 0
        for card in page_extraction.cards:
            bbox = matched.get(card.sku)
            if not bbox:
                summary["cards_unchanged"] += 1
                continue
            if card.photo_bbox_px and not force:
                # Already set by a prior run; don't overwrite without --force
                summary["cards_unchanged"] += 1
                continue
            card.photo_bbox_px = list(bbox)
            page_matched += 1

        if page_matched > 0:
            save_page_sidecar(pdf_path, page_no, page_extraction)
            summary["pages_updated"] += 1
            summary["cards_matched"] += page_matched
            if verbose:
                print(f"[backfill] page {page_no}: matched {page_matched} "
                      f"card(s)", file=sys.stderr)
    if verbose:
        print(
            f"[backfill] DONE  scanned={summary['pages_scanned']}  "
            f"updated={summary['pages_updated']}  "
            f"matched={summary['cards_matched']}  "
            f"unchanged={summary['cards_unchanged']}",
            file=sys.stderr,
        )
    return summary
