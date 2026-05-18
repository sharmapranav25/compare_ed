"""PDF page rendering helpers — lifted from phase3/annotate.py.

Renders any page of a PDF to PNG bytes at a consistent DPI. Also provides a
base64 image block builder for Anthropic API messages.

DPI=180 is the proven sweet spot: legible text for VLM card detection without
blowing past the API's 5 MB base64 limit on full-page renders.
"""
from __future__ import annotations

import base64
import io

import pypdfium2 as pdfium

RENDER_DPI = 180


def render_page_image(pdf: pdfium.PdfDocument, page_no_1: int, dpi: int = RENDER_DPI) -> bytes:
    """Render a 1-indexed PDF page to PNG bytes."""
    page = pdf[page_no_1 - 1]
    bitmap = page.render(scale=dpi / 72.0)
    img = bitmap.to_pil()
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def render_page_image_capped(
    pdf: pdfium.PdfDocument, page_no_1: int, dpi: int = RENDER_DPI, max_bytes: int = 3_700_000
) -> bytes:
    """Render a page, downscaling if the raw PNG would exceed the Anthropic 5 MB
    base64 limit (raw 3.7 MB → ~5 MB base64).

    Picks up the lesson from the Hanger Clinic match_photos crash: high-DPI
    catalogs can produce single-page renders past the limit. Defensive shrink.
    """
    page = pdf[page_no_1 - 1]
    bitmap = page.render(scale=dpi / 72.0)
    pil = bitmap.to_pil()
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    data = buf.getvalue()
    while len(data) > max_bytes and min(pil.size) > 256:
        new_size = (max(1, int(pil.size[0] * 0.75)), max(1, int(pil.size[1] * 0.75)))
        pil = pil.resize(new_size)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        data = buf.getvalue()
    return data


# Anthropic Vision recommends max long-edge of ~1568px for best results.
# Larger images are auto-downscaled by the API, which produces ambiguous
# coordinate spaces in returned bboxes. We pin our own size so the bboxes
# returned by the VLM are in a known coordinate system we control.
VLM_MAX_LONG_EDGE_PX = 1568


def render_page_for_vlm(
    pdf: pdfium.PdfDocument,
    page_no_1: int,
    max_long_edge: int = VLM_MAX_LONG_EDGE_PX,
) -> tuple[bytes, int, int]:
    """Render a page at a fixed max long-edge size suitable for VLM input.

    Returns (png_bytes, width_px, height_px). The dimensions returned are the
    EXACT pixel dims of the PNG that gets sent to the API, so the VLM's bbox
    coordinates can be interpreted unambiguously.
    """
    page = pdf[page_no_1 - 1]
    # PDF point size -> target pixel size at max_long_edge for the long side
    pt_w = page.get_width()
    pt_h = page.get_height()
    long_pt = max(pt_w, pt_h)
    scale = max_long_edge / long_pt
    bitmap = page.render(scale=scale)
    pil = bitmap.to_pil()
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue(), pil.size[0], pil.size[1]


def b64_image_block(png_bytes: bytes, *, cache: bool = False) -> dict:
    """Build an Anthropic API image content block from PNG bytes.

    Set cache=True to mark this block as ephemeral-cacheable across calls.
    Useful when the same page image is referenced from multiple sequential
    calls (e.g. card detection + field extraction).
    """
    block: dict = {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": base64.standard_b64encode(png_bytes).decode(),
        },
    }
    if cache:
        block["cache_control"] = {"type": "ephemeral"}
    return block
