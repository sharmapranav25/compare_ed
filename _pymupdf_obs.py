"""PyMuPDF observation gathering for one PDF page.

Native-PDF-object enumeration: selectable text words + their bboxes, and
the rectangles of every embedded image XObject. This is the exact (vs.
heuristic) detector running alongside YOLO. When a PDF was generated
from an InDesign / typesetting workflow, image rects + xrefs come back
verbatim from the page content stream — no inference involved.

Output coordinates are in PIXELS of the matching full-res render so the
matcher can compare with YOLO boxes without bookkeeping the scale at
every call site. `render_scale` is also returned for callers that
need to round-trip back to PDF points.

Public surface:
  observations(pdf, page_no, dpi=300) -> dict
"""

from __future__ import annotations

import fitz

DEFAULT_DPI = 300


def _rect_to_px(rect, scale: float) -> list[float]:
    """fitz.Rect (PDF points) -> [x1, y1, x2, y2] pixels."""
    return [rect.x0 * scale, rect.y0 * scale,
            rect.x1 * scale, rect.y1 * scale]


def observations(pdf: fitz.Document, page_no: int,
                 dpi: int = DEFAULT_DPI) -> dict:
    """Collect words + image XObject rects on `page_no` (1-indexed).

    Returns:
      {
        "page_no": N,
        "page_size_pt":  [w_pt, h_pt],
        "page_size_px":  [w_px, h_px],
        "render_scale":  dpi / 72.0,
        "words": [
          {"text": "A24021", "bbox_px": [x1,y1,x2,y2],
           "block": 0, "line": 0, "word": 0}, ...
        ],
        "image_rects": [
          {"xref": 117, "bbox_px": [x1,y1,x2,y2],
           "smask": 0, "width": 1200, "height": 1200, "bpc": 8}, ...
        ]
      }

    Words: one entry per "word" as fitz tokenizes (whitespace + line
    splits — PyMuPDF's own segmentation). Bbox is the rect of that
    word as it appears on the page. SKU strings often span multiple
    words; downstream localization stitches adjacent words together.

    Image rects: one entry per (xref, rect) pairing — the same xref
    can appear multiple times on a page if reused, so the result is
    page_size_px-flat, not xref-deduped.
    """
    page = pdf[page_no - 1]
    scale = dpi / 72.0
    w_pt, h_pt = page.rect.width, page.rect.height
    w_px, h_px = w_pt * scale, h_pt * scale

    words_out: list[dict] = []
    # page.get_text("words") -> (x0, y0, x1, y1, "word", block_no, line_no, word_no)
    for tup in page.get_text("words"):
        x0, y0, x1, y1, text, block_no, line_no, word_no = tup
        if not text:
            continue
        words_out.append({
            "text": text,
            "bbox_px": [x0 * scale, y0 * scale, x1 * scale, y1 * scale],
            "block": int(block_no),
            "line": int(line_no),
            "word": int(word_no),
        })

    images_out: list[dict] = []
    # page.get_images returns: (xref, smask, w, h, bpc, colorspace, ...)
    for img in page.get_images(full=True):
        xref = int(img[0])
        smask = int(img[1]) if len(img) > 1 else 0
        iw = int(img[2]) if len(img) > 2 else 0
        ih = int(img[3]) if len(img) > 3 else 0
        bpc = int(img[4]) if len(img) > 4 else 0
        try:
            rects = page.get_image_rects(xref)
        except Exception:  # noqa: BLE001
            rects = []
        for rect in rects:
            images_out.append({
                "xref": xref,
                "bbox_px": _rect_to_px(rect, scale),
                "smask": smask,
                "width": iw,
                "height": ih,
                "bpc": bpc,
            })

    return {
        "page_no": page_no,
        "page_size_pt": [float(w_pt), float(h_pt)],
        "page_size_px": [float(w_px), float(h_px)],
        "render_scale": float(scale),
        "words": words_out,
        "image_rects": images_out,
    }


def find_sku_word_bbox(words: list[dict], sku: str) -> list[float] | None:
    """Locate a SKU string in the word list. Returns the union bbox of
    consecutive matching words, or None if not found.

    Matching strategy:
      1. Normalize both sides by stripping whitespace + lowering case.
      2. Try the SKU as a single word first (most common).
      3. Else, walk the word list and stitch consecutive words on the
         same (block, line) until their concatenation == the normalized
         SKU. Allow at most 4 word-pieces — wider stitches are usually
         spurious matches.
    """
    if not sku:
        return None
    target = "".join(sku.split()).lower()
    if not target:
        return None

    # Step 1: single-word match
    for w in words:
        t = "".join((w["text"] or "").split()).lower()
        if t == target:
            return list(w["bbox_px"])

    # Step 2: stitched-on-same-line match
    by_line: dict[tuple[int, int], list[dict]] = {}
    for w in words:
        by_line.setdefault((w["block"], w["line"]), []).append(w)
    for line_words in by_line.values():
        line_words.sort(key=lambda w: w["word"])
        n = len(line_words)
        for start in range(n):
            joined = ""
            for end in range(start, min(start + 4, n)):
                joined += "".join((line_words[end]["text"] or "").split()).lower()
                if joined == target:
                    bbs = [line_words[k]["bbox_px"] for k in range(start, end + 1)]
                    x1 = min(b[0] for b in bbs)
                    y1 = min(b[1] for b in bbs)
                    x2 = max(b[2] for b in bbs)
                    y2 = max(b[3] for b in bbs)
                    return [x1, y1, x2, y2]
                if not target.startswith(joined):
                    break
    return None
