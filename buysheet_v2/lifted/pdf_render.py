"""PDF page rendering helpers — lifted from phase3/annotate.py.

Renders any page of a PDF to PNG bytes at a consistent DPI. Also provides a
base64 image block builder for Anthropic API messages.

DPI=180 is the proven sweet spot: legible text for VLM card detection without
blowing past the API's 5 MB base64 limit on full-page renders.

For the SKU→box matcher specifically we render a SEPARATE hi-res copy at
MATCHER_MAX_LONG_EDGE_PX (3000) so dense grid captions are legible. To stay
under Anthropic's ~3.6 MB raw-image cap (5 MB base64) at that resolution we
walk ENCODE_STRATEGIES — PNG first (lossless, crisp for tiny text), then
JPEG-q85 (handles photo-heavy pages PNG cannot compress). Ported from
parent repo's _render.py.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Optional

import pypdfium2 as pdfium
from PIL import Image

RENDER_DPI = 180

# Hi-res render budget used ONLY by the matcher annotation path. The
# coordinate-frame contract is: lo-res 1568 px space stays canonical
# everywhere else (extract, verify, write, photo_vlm, judge). The hi-res
# render exists purely so Sonnet can read tiny captions under YOLO's
# numbered boxes; bboxes are scaled INTO hi-res inside annotate_page_at_scale
# and never escape that function.
MATCHER_MAX_LONG_EDGE_PX = 3000

# Encode cascade — same as parent _render.py. Each strategy is tried in
# order, with up to MAX_DOWNSCALE_PASSES shrink passes per strategy. PNG
# wins on text-heavy pages (lossless); JPEG-q85 catches photo-heavy decks
# PNG can't compress.
MAX_IMAGE_BYTES = 3_600_000
MAX_IMAGE_DIM = 8000
MAX_DOWNSCALE_PASSES = 4
ENCODE_STRATEGIES: tuple[dict, ...] = (
    {"format": "PNG",  "suffix": ".png", "save_kwargs": {"optimize": True}},
    {"format": "JPEG", "suffix": ".jpg",
     "save_kwargs": {"quality": 85, "optimize": True}},
)


class ImageTooLargeError(RuntimeError):
    """No encoder strategy produced an image under MAX_IMAGE_BYTES."""


def _fit_within_dim(img: Image.Image, max_dim: int) -> Image.Image:
    w, h = img.size
    if w <= max_dim and h <= max_dim:
        return img
    s = max_dim / float(max(w, h))
    return img.resize((max(1, int(w * s)), max(1, int(h * s))))


def media_type_for(path: Path) -> str:
    """Map a render file's suffix to its Anthropic API media_type."""
    s = path.suffix.lower()
    if s in (".jpg", ".jpeg"):
        return "image/jpeg"
    return "image/png"


def save_with_cascade(img: Image.Image, base_out_path: Path) -> Path:
    """Save a PIL image walking ENCODE_STRATEGIES; first to fit wins.

    `base_out_path` is the path WITHOUT extension. The returned Path has the
    suffix of the winning strategy (.png or .jpg). Raises ImageTooLargeError
    if no strategy fits within the byte cap after MAX_DOWNSCALE_PASSES
    shrink passes. Idempotent — caller can re-invoke after deleting the
    output.
    """
    img = _fit_within_dim(img, MAX_IMAGE_DIM)
    tried: list[str] = []
    for strategy in ENCODE_STRATEGIES:
        candidate = img
        for pass_no in range(MAX_DOWNSCALE_PASSES + 1):
            buf = io.BytesIO()
            candidate.save(buf, format=strategy["format"],
                           **strategy["save_kwargs"])
            if buf.tell() <= MAX_IMAGE_BYTES:
                out = base_out_path.with_suffix(strategy["suffix"])
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(buf.getvalue())
                return out
            if pass_no == MAX_DOWNSCALE_PASSES:
                tried.append(
                    f"{strategy['format']}({buf.tell()}B@pass{pass_no})"
                )
                break
            w, h = candidate.size
            candidate = candidate.resize(
                (max(1, int(w * 0.75)), max(1, int(h * 0.75)))
            )
    raise ImageTooLargeError(
        f"{base_out_path.name}: no encoder strategy fit under "
        f"{MAX_IMAGE_BYTES} bytes (tried {', '.join(tried)})"
    )


def render_page_for_matcher(
    pdf: pdfium.PdfDocument,
    page_no_1: int,
    max_long_edge: int = MATCHER_MAX_LONG_EDGE_PX,
) -> tuple[Image.Image, int, int]:
    """Render a page at hi-res for the matcher annotation path.

    Returns (PIL.Image, width_px, height_px) — caller passes the Image to
    annotate_page_at_scale which draws bboxes (scaled from 1568 space) and
    persists via save_with_cascade. The image is NOT encoded to bytes here
    so the bbox-drawing pass stays in PIL space — single round-trip.
    """
    page = pdf[page_no_1 - 1]
    pt_w = page.get_width()
    pt_h = page.get_height()
    long_pt = max(pt_w, pt_h)
    scale = max_long_edge / long_pt
    bitmap = page.render(scale=scale)
    pil = bitmap.to_pil().convert("RGB")
    return pil, pil.size[0], pil.size[1]


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


def b64_image_block_from_path(path: Path, *, cache: bool = False) -> dict:
    """Read an image file and build an Anthropic API content block.

    Picks media_type from the file's suffix (".jpg"/".jpeg" → image/jpeg,
    everything else → image/png). Used for the hi-res annotated PNG path
    where the cascade may have produced a JPEG fallback.
    """
    block: dict = {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type_for(path),
            "data": base64.standard_b64encode(path.read_bytes()).decode(),
        },
    }
    if cache:
        block["cache_control"] = {"type": "ephemeral"}
    return block
