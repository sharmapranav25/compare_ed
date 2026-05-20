"""Photo embedding helpers — lifted from phase5/populate.py.

The two functions that survive:
- crop_to_silo: blob-based isolation of the dominant product silhouette
  in a card bitmap (strips PPT chrome / accessory thumbnails).
- embed_photo: openpyxl write of a PNG into a workbook cell anchored as
  a TwoCellAnchor (editAs="twoCell"). Both the top-left AND bottom-right
  corners pin to cells, so the image moves+resizes with the cell instead
  of floating above it. Excel respects this directly. Google Sheets still
  imports as a floating image — run apps_script/fix_images.gs after
  import to convert those to true in-cell CellImage objects.

Previously this used OneCellAnchor which only pinned the top-left corner,
producing images that visibly hovered over the cell.
"""
from __future__ import annotations

import io
from pathlib import Path

from openpyxl.comments import Comment
from openpyxl.drawing.image import Image as XLImage
from openpyxl.drawing.spreadsheet_drawing import (
    AnchorMarker, TwoCellAnchor,
)
from PIL import Image as PILImage

try:
    import numpy as _np
    from scipy import ndimage as _ndimage

    _SILO_CROP_AVAILABLE = True
except ImportError:
    _np = None
    _ndimage = None
    _SILO_CROP_AVAILABLE = False

PHOTO_MAX_WIDTH_PX = 84
PHOTO_MAX_HEIGHT_PX = 76


def crop_to_silo(
    img: PILImage.Image,
    dilate_iter: int = 2,
    pad_px: int = 4,
    min_blob_pixels: int = 200,
) -> PILImage.Image:
    """Crop to the dominant product silhouette in a card bitmap.

    Finds the largest connected non-white blob (the shoe), weighted by mass ×
    density and penalized for extreme aspect ratios. Drops PPT navigator chrome
    and accessory thumbnails. No-op when scipy isn't available.
    """
    if not _SILO_CROP_AVAILABLE:
        return img
    arr = _np.array(img.convert("L"))
    h, w = arr.shape
    mask = arr < 240
    if int(mask.sum()) < min_blob_pixels:
        return img
    dilated = _ndimage.binary_dilation(mask, iterations=dilate_iter)
    labeled, n = _ndimage.label(dilated)
    if n == 0:
        return img

    best = None
    best_score = -1.0
    for lbl in range(1, n + 1):
        ys, xs = _np.where(labeled == lbl)
        if len(ys) < min_blob_pixels:
            continue
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        bb_w = x1 - x0 + 1
        bb_h = y1 - y0 + 1
        aspect = max(bb_w, bb_h) / max(1, min(bb_w, bb_h))
        if aspect > 4.0:
            continue
        mass = int(mask[y0 : y1 + 1, x0 : x1 + 1].sum())
        density = mass / max(1, bb_w * bb_h)
        score = mass * density
        if score > best_score:
            best_score = score
            best = (x0, y0, x1, y1)
    if best is None:
        return img
    x0, y0, x1, y1 = best
    x0 = max(0, x0 - pad_px)
    y0 = max(0, y0 - pad_px)
    x1 = min(w - 1, x1 + pad_px)
    y1 = min(h - 1, y1 + pad_px)
    if x1 - x0 < 10 or y1 - y0 < 10:
        return img
    return img.crop((x0, y0, x1 + 1, y1 + 1))


def embed_photo(
    ws,
    row: int,
    col: int,
    image_source: PILImage.Image | Path | bytes,
    comment_text: str | None = None,
) -> bool:
    """Embed a photo into a workbook cell.

    image_source can be a PIL image, a path to a PNG, or raw PNG bytes.
    Downsizes to PHOTO_MAX_WIDTH_PX × PHOTO_MAX_HEIGHT_PX and isolates the
    silhouette before write. Returns True on success.
    """
    try:
        if isinstance(image_source, PILImage.Image):
            img = image_source.convert("RGB")
        elif isinstance(image_source, Path):
            img = PILImage.open(image_source).convert("RGB")
        elif isinstance(image_source, (bytes, bytearray)):
            img = PILImage.open(io.BytesIO(image_source)).convert("RGB")
        else:
            return False
    except Exception:
        return False

    img = crop_to_silo(img)
    img.thumbnail((PHOTO_MAX_WIDTH_PX, PHOTO_MAX_HEIGHT_PX), PILImage.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    cell = ws.cell(row, col)
    xlimg = XLImage(buf)
    # TwoCellAnchor with editAs="twoCell" pins BOTH corners to cells —
    # the image is owned by the cell rectangle (row,col) → (row+1,col+1)
    # and moves + resizes with the cell. OneCellAnchor (the previous
    # behavior) only pinned the top-left so the image visibly hovered.
    # AnchorMarker uses 0-indexed coords; openpyxl row/col are 1-indexed.
    from_marker = AnchorMarker(col=col - 1, colOff=0, row=row - 1, rowOff=0)
    to_marker = AnchorMarker(col=col, colOff=0, row=row, rowOff=0)
    xlimg.anchor = TwoCellAnchor(
        editAs="twoCell", _from=from_marker, to=to_marker,
    )
    ws.add_image(xlimg)
    if comment_text:
        cell.comment = Comment(comment_text, "Kith Buysheet Agent v2")
    return True
