"""Shared PDF page → image render helper with Anthropic image-size guard.

Anthropic's API caps a single image at 5 MiB of base64 payload (~33% larger
than raw bytes). That puts the safe raw-image ceiling at ~3.6 MB. Catalog
pages with full-bleed photography often exceed this at 300 DPI; photo-heavy
presentation decks (1920×1080 marketing slides) can't fit in PNG at any
sane DPI because PNG compresses photographs terribly.

`render_page_safe()` walks ENCODE_STRATEGIES in order, applying the
progressive downscale guard within each strategy. The first strategy that
fits under MAX_IMAGE_BYTES wins; its file extension determines the API
media_type the caller will use. Adding a new format (WEBP, AVIF, …) is one
row in ENCODE_STRATEGIES — call-site code doesn't change because the
caller reads `.suffix` off the returned path.

Existing `.pages/` directories with .png files from prior runs remain
valid cache: `find_cached_render` looks for either `.png` or `.jpg`.

Ported from the legacy phase1/reconcile.py:render_page_b64 logic; extended
2026-05-18 to handle photo-heavy decks that PNG cannot compress.
"""

from __future__ import annotations

import io
from pathlib import Path

import fitz
from PIL import Image

MAX_IMAGE_BYTES = 3_600_000  # raw image cap to stay under the 5 MiB base64 cap
MAX_IMAGE_DIM = 8000         # Anthropic also rejects either side > 8000 px
MAX_DOWNSCALE_PASSES = 4
DEFAULT_DPI = 300

# Encoders are tried in order; the first to fit (after up to
# MAX_DOWNSCALE_PASSES shrink passes) wins. PNG goes first because it's
# lossless and crisper for text/SKU/price OCR; JPEG-q85 handles photo-
# heavy pages PNG can't compress. Add a new format by appending a row —
# no call-site changes needed.
ENCODE_STRATEGIES: tuple[dict, ...] = (
    {"format": "PNG",  "suffix": ".png", "save_kwargs": {"optimize": True}},
    {"format": "JPEG", "suffix": ".jpg",
     "save_kwargs": {"quality": 85, "optimize": True}},
)

CACHE_SUFFIXES = tuple(s["suffix"] for s in ENCODE_STRATEGIES)


class ImageTooLargeError(RuntimeError):
    """No encoder strategy produced an image under MAX_IMAGE_BYTES."""


def _fit_within_dim(img: Image.Image, max_dim: int) -> Image.Image:
    w, h = img.size
    if w <= max_dim and h <= max_dim:
        return img
    s = max_dim / float(max(w, h))
    return img.resize((max(1, int(w * s)), max(1, int(h * s))))


def find_cached_render(pages_dir: Path, page_no: int) -> Path | None:
    """Return the existing rendered file for this page if any encoder
    strategy has already produced one — tolerates both .png (legacy and
    PNG-strategy winners) and .jpg (JPEG fallback winners)."""
    for ext in CACHE_SUFFIXES:
        p = pages_dir / f"{page_no:02d}{ext}"
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def media_type_for(path: Path) -> str:
    """Map a render file's suffix to its Anthropic API media_type."""
    s = path.suffix.lower()
    if s in (".jpg", ".jpeg"):
        return "image/jpeg"
    return "image/png"


def render_page_safe(pdf: fitz.Document, page_no: int, pages_dir: Path,
                     dpi: int = DEFAULT_DPI) -> Path:
    """Render page_no (1-indexed) to <pages_dir>/<NN>.<ext>.

    Tries each ENCODE_STRATEGIES entry in order, with up to
    MAX_DOWNSCALE_PASSES shrink passes per strategy. Returns the actual
    written path (caller reads .suffix for the API media_type).
    """
    page = pdf[page_no - 1]
    scale = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    base_img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    base_img = _fit_within_dim(base_img, MAX_IMAGE_DIM)

    tried: list[str] = []
    for strategy in ENCODE_STRATEGIES:
        img = base_img
        for pass_no in range(MAX_DOWNSCALE_PASSES + 1):
            buf = io.BytesIO()
            img.save(buf, format=strategy["format"], **strategy["save_kwargs"])
            if buf.tell() <= MAX_IMAGE_BYTES:
                out = pages_dir / f"{page_no:02d}{strategy['suffix']}"
                out.write_bytes(buf.getvalue())
                return out
            if pass_no == MAX_DOWNSCALE_PASSES:
                tried.append(f"{strategy['format']}({buf.tell()}B@pass{pass_no})")
                break
            w, h = img.size
            img = img.resize((max(1, int(w * 0.75)), max(1, int(h * 0.75))))

    raise ImageTooLargeError(
        f"page {page_no}: no encoder strategy fit under "
        f"{MAX_IMAGE_BYTES} bytes (tried {', '.join(tried)})"
    )
