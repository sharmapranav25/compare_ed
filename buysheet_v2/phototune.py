"""Deterministic image binding: SKU pixel anchor + layout-aware photo region.

The VLM's card_bbox / photo_bbox is unreliable on dense grids — it can pair
the right SKU label with the wrong visual card. PyMuPDF's text-search returns
the EXACT pixel rect of any text on a page (when a text layer exists), giving
us a ground-truth anchor for every SKU.

Pipeline:
  1. find_sku_pixel_rect: PyMuPDF text-search → pixel coords of SKU on page
  2. compute_photo_region: heuristic photo bbox from SKU pos + layout type
  3. reconcile_photo_bbox: pick VLM bbox if it sanity-matches, else deterministic
  4. pad_bbox: 15% margin before cropping (prevents silhouette cut-off)

For image-only pages (PyMuPDF returns no rects), the VLM's photo_bbox is the
only signal — accept it as-is.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF


# ---------------------------------------------------------------------------- text-search

@lru_cache(maxsize=4)
def _open_doc(pdf_path: str) -> fitz.Document:
    """Cache the PyMuPDF document handle — caller still owns close()."""
    return fitz.open(pdf_path)


def find_sku_pixel_rect(
    pdf_path: Path, page_no_1: int, sku: str, render_w: int, render_h: int
) -> Optional[tuple[int, int, int, int]]:
    """PyMuPDF text-search for SKU; translate result to render-pixel coords.

    Returns (x1, y1, x2, y2) in TL-origin pixel coords matching the VLM's coord
    system (page rendered at long_edge=max(render_w, render_h)).

    None if SKU not found (image-only pages, or text-layer doesn't carry SKU).
    """
    try:
        doc = fitz.open(str(pdf_path))
    except Exception:
        return None
    try:
        page = doc.load_page(page_no_1 - 1)
        pt_w, pt_h = page.rect.width, page.rect.height
        sx = render_w / pt_w
        sy = render_h / pt_h
        # Direct text-search
        rects = page.search_for(sku)
        if not rects:
            # Try with whitespace tolerance — PyMuPDF doesn't natively, so retry
            # a few permutations
            for variant in (sku.replace("-", " - "), sku.replace("-", "")):
                if variant != sku:
                    rects = page.search_for(variant)
                    if rects:
                        break
        if not rects:
            return None
        r = rects[0]  # first occurrence; SKUs are typically unique per page
        return (
            int(r.x0 * sx),
            int(r.y0 * sy),
            int(r.x1 * sx),
            int(r.y1 * sy),
        )
    finally:
        doc.close()


# ---------------------------------------------------------------------------- layout-aware photo region

def cluster_sku_rows(
    skus_with_rects: list[tuple[str, tuple[int, int, int, int]]],
    y_tolerance: int = 20,
) -> list[list[tuple[str, tuple[int, int, int, int]]]]:
    """Group SKUs into rows by approximate y position. Returns rows top-to-bottom,
    each row sorted left-to-right.
    """
    if not skus_with_rects:
        return []
    sorted_by_y = sorted(skus_with_rects, key=lambda kv: kv[1][1])
    rows: list[list[tuple[str, tuple[int, int, int, int]]]] = []
    cur_row: list[tuple[str, tuple[int, int, int, int]]] = []
    cur_y: Optional[int] = None
    for sku, rect in sorted_by_y:
        y = rect[1]
        if cur_y is None or abs(y - cur_y) <= y_tolerance:
            cur_row.append((sku, rect))
            if cur_y is None:
                cur_y = y
        else:
            rows.append(sorted(cur_row, key=lambda kv: kv[1][0]))
            cur_row = [(sku, rect)]
            cur_y = y
    if cur_row:
        rows.append(sorted(cur_row, key=lambda kv: kv[1][0]))
    return rows


def estimate_column_width(skus_with_rects: list[tuple[str, tuple[int, int, int, int]]]) -> Optional[int]:
    """Median pixel distance between adjacent SKU label x-centers on the SAME row."""
    rows = cluster_sku_rows(skus_with_rects)
    deltas: list[float] = []
    for row in rows:
        centers = [(r[1][0] + r[1][2]) / 2 for r in row]
        for i in range(len(centers) - 1):
            d = centers[i + 1] - centers[i]
            if 30 < d < 400:
                deltas.append(d)
    if not deltas:
        return None
    deltas.sort()
    return int(deltas[len(deltas) // 2])


def compute_grid_photo_bboxes(
    sku_rects: dict[str, tuple[int, int, int, int]],
    page_w: int,
    page_h: int,
    *,
    text_block_height_pct: float = 0.45,
) -> dict[str, tuple[int, int, int, int]]:
    """For a grid/vertical-card layout: compute every card's photo bbox from
    PyMuPDF SKU pixel positions + sibling row/col spacing.

    Logic per row N:
      - photo Y-range = full vertical space between row N-1's text block and
        row N's SKU label
      - photo X-range = column_width centered on each SKU's x-center
      - text_block_height_pct controls how much of the previous row's vertical
        space is reserved for text below the previous SKU (color/date/price)

    This is the architecturally correct image binding for grid catalogs:
    every photo is computed deterministically from the SKU's anchor + row
    geometry. No VLM bbox needed.
    """
    rows = cluster_sku_rows(list(sku_rects.items()))
    if not rows:
        return {}

    # Use median col_w across all rows for stability
    col_w_global = estimate_column_width(list(sku_rects.items())) or 100

    out: dict[str, tuple[int, int, int, int]] = {}
    for ri, row in enumerate(rows):
        # Use the global median col_w. Per-row median can blow up when a row
        # has sparse products (e.g. last row with only 4 products spaced 200px
        # apart instead of the typical 100px column width).
        # Per-row min spacing is a sanity check — if it's significantly smaller
        # than global, use that instead (catches denser-than-average rows).
        col_w = col_w_global
        if len(row) >= 2:
            centers = [(r[1][0] + r[1][2]) / 2 for r in row]
            row_min_delta = min(
                centers[i + 1] - centers[i] for i in range(len(centers) - 1)
            )
            if 30 < row_min_delta < col_w_global:
                col_w = int(row_min_delta)

        row_sku_top = min(r[1][1] for r in row)
        # Y-top: just below the previous row's text block (or page top for row 0)
        if ri > 0:
            prev_row = rows[ri - 1]
            prev_sku_top = min(r[1][1] for r in prev_row)
            row_height = row_sku_top - prev_sku_top
            # Previous row's text block (SKU + color + date + price) typically
            # takes ~45% of the row spacing. Photo for THIS row fills the rest.
            prev_text_bottom = prev_sku_top + int(row_height * text_block_height_pct)
            photo_y_top = max(0, prev_text_bottom + 4)
        else:
            photo_y_top = 0

        # The card layout typically has: photo → description (~15px) → SKU.
        # Pull photo_y_bottom up by ~20px to leave room for the description line.
        photo_y_bottom = max(photo_y_top + 20, row_sku_top - 22)

        for sku, rect in row:
            sku_cx = (rect[0] + rect[2]) // 2
            x1 = max(0, sku_cx - col_w // 2)
            x2 = min(page_w, sku_cx + col_w // 2)
            out[sku] = (x1, photo_y_top, x2, photo_y_bottom)
    return out


def compute_photo_region(
    sku_rect: tuple[int, int, int, int],
    layout_type: str,
    page_w: int,
    page_h: int,
    card_bbox: Optional[list[int]] = None,
    column_width: Optional[int] = None,
) -> Optional[tuple[int, int, int, int]]:
    """Single-SKU fallback: rough photo bbox when grid layout info isn't available.

    For grid layouts, prefer compute_grid_photo_bboxes which uses row context.
    This function is the single-card fallback (e.g. one SKU on a sparse page).
    """
    sx1, sy1, sx2, sy2 = sku_rect
    sku_cx = (sx1 + sx2) // 2

    if layout_type in ("grid", "vertical_card"):
        if column_width and column_width > 20:
            col_w = column_width
        elif card_bbox:
            col_w = card_bbox[2] - card_bbox[0]
        else:
            col_w = (sx2 - sx1) * 2.5
        col_x1 = max(0, sku_cx - col_w // 2)
        col_x2 = min(page_w, sku_cx + col_w // 2)
        photo_y2 = max(0, sy1 - 4)
        photo_y1 = max(0, photo_y2 - int(col_w * 1.4))  # taller than wide is OK
        return (col_x1, photo_y1, col_x2, photo_y2)

    elif layout_type == "spec_panel":
        if card_bbox:
            cb_x1, cb_y1, cb_x2, cb_y2 = card_bbox
            photo_height = int((cb_y2 - cb_y1) * 0.7)
            return (cb_x1, cb_y1, cb_x2, cb_y1 + photo_height)
        return None

    elif layout_type == "lookbook":
        if card_bbox:
            return tuple(card_bbox)
        return None

    return None


# ---------------------------------------------------------------------------- reconciliation + padding

def reconcile_photo_bbox(
    vlm_bbox: Optional[list[int]],
    deterministic_bbox: Optional[tuple[int, int, int, int]],
    card_bbox: Optional[list[int]] = None,
    *,
    column_width: Optional[int] = None,
) -> Optional[tuple[int, int, int, int]]:
    """Pick the best photo bbox by comparing CENTER positions.

    The VLM bbox is accepted ONLY when its center is close to the deterministic
    center (within col_width/3 horizontally, col_width/2 vertically). Otherwise
    the deterministic bbox wins.

    Bbox-membership-with-slop was too lenient — a VLM bbox shifted by 44px in
    a 100px-wide column was accepted because slop extended the region 20% on
    each side. Center-distance comparison is robust to bbox size differences
    (deterministic is column-wide, VLM may be silhouette-tight) while still
    catching real cross-column mis-attribution.
    """
    if deterministic_bbox is None:
        if vlm_bbox is None:
            return None
        return tuple(vlm_bbox)
    if vlm_bbox is None:
        return deterministic_bbox

    vlm_cx = (vlm_bbox[0] + vlm_bbox[2]) / 2
    vlm_cy = (vlm_bbox[1] + vlm_bbox[3]) / 2
    dx1, dy1, dx2, dy2 = deterministic_bbox
    det_cx = (dx1 + dx2) / 2
    det_cy = (dy1 + dy2) / 2
    det_w = dx2 - dx1
    det_h = dy2 - dy1

    # Tolerances based on column width (preferred) or det_w fallback
    cw = column_width or det_w
    x_tol = max(15, cw / 3)
    y_tol = max(20, det_h / 2)

    if abs(vlm_cx - det_cx) <= x_tol and abs(vlm_cy - det_cy) <= y_tol:
        return tuple(vlm_bbox)
    return deterministic_bbox


def pad_bbox(
    bbox: tuple[int, int, int, int],
    page_w: int,
    page_h: int,
    pct: float = 0.15,
) -> tuple[int, int, int, int]:
    """Pad bbox by pct on all sides, clamped to page bounds.

    Prevents tight crops from cutting off shoe silhouettes (the Converse issue).
    """
    x1, y1, x2, y2 = bbox
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    pad_x = int(w * pct)
    pad_y = int(h * pct)
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(page_w, x2 + pad_x),
        min(page_h, y2 + pad_y),
    )


# ---------------------------------------------------------------------------- per-page batch helper

def resolve_page_photo_bboxes(
    pdf_path: Path,
    page_no: int,
    cards: list,  # list[ProductCard]
    layout_type: str,
    page_w: int,
    page_h: int,
) -> dict[str, tuple[int, int, int, int]]:
    """One-shot photo-bbox resolution for every card on a page.

    For grid/vertical_card layouts: row-context geometry from compute_grid_photo_bboxes
    (PyMuPDF SKU anchor + row/col spacing — fully deterministic).
    For spec_panel/lookbook: per-card heuristic via compute_photo_region.

    Returns {sku: final_photo_bbox} in page-pixel coords, padded 15%.
    Cards whose SKU isn't found in the text layer are omitted (image-only page).
    """
    # Step 1: text-search every SKU on the page
    sku_rects: dict[str, tuple[int, int, int, int]] = {}
    for c in cards:
        rect = find_sku_pixel_rect(pdf_path, page_no, c.sku, page_w, page_h)
        if rect:
            sku_rects[c.sku] = rect

    out: dict[str, tuple[int, int, int, int]] = {}

    if layout_type in ("grid", "vertical_card") and len(sku_rects) >= 2:
        # Use row-context geometry — robust to dense grids
        deterministic = compute_grid_photo_bboxes(sku_rects, page_w, page_h)
        col_w = estimate_column_width(list(sku_rects.items()))
        # Reconcile each VLM bbox against the deterministic anchor
        for c in cards:
            det = deterministic.get(c.sku)
            if det is None:
                continue
            final = reconcile_photo_bbox(c.photo_bbox_px, det, c.card_bbox_px, column_width=col_w)
            if final:
                out[c.sku] = pad_bbox(final, page_w, page_h, pct=0.15)
    else:
        # Single-SKU / spec_panel / lookbook fallback
        col_w = estimate_column_width(list(sku_rects.items())) if sku_rects else None
        for c in cards:
            sku_rect = sku_rects.get(c.sku)
            det_bbox = None
            if sku_rect:
                det_bbox = compute_photo_region(
                    sku_rect, layout_type, page_w, page_h,
                    card_bbox=c.card_bbox_px, column_width=col_w,
                )
            final = reconcile_photo_bbox(c.photo_bbox_px, det_bbox, c.card_bbox_px, column_width=col_w)
            if final:
                out[c.sku] = pad_bbox(final, page_w, page_h, pct=0.15)
    return out
