"""Shared PDF page → PNG render helper with Anthropic image-size guard.

Anthropic's API caps a single image at 5 MiB of base64 payload (~33% larger
than raw bytes). That puts the safe raw-PNG ceiling at ~3.6 MB. Catalog
pages with full-bleed photography often exceed this at 300 DPI.

`render_page_png_safe()` renders at the requested DPI, checks the encoded
PNG size, and progressively downscales the PIL image by 75% per pass up to
4 passes (final = ~0.32× original area) until it fits. If it still doesn't
fit after 4 passes, raises ImageTooLargeError so the caller can flag the
page in the per-page JSON instead of silently letting the LLM call fail.

Ported from the legacy phase1/reconcile.py:render_page_b64 logic.
"""

from __future__ import annotations

import io
from pathlib import Path

import fitz
from PIL import Image

MAX_IMAGE_BYTES = 3_600_000  # raw PNG cap to stay under the 5 MiB base64 cap
MAX_DOWNSCALE_PASSES = 4
DEFAULT_DPI = 300


class ImageTooLargeError(RuntimeError):
    """Page image exceeded MAX_IMAGE_BYTES even after MAX_DOWNSCALE_PASSES."""


def render_page_png_safe(pdf: fitz.Document, page_no: int, out_path: Path,
                         dpi: int = DEFAULT_DPI) -> Path:
    """Render page_no (1-indexed) at `dpi` to `out_path` (PNG).

    Downscales 75% per pass up to MAX_DOWNSCALE_PASSES if the encoded PNG
    exceeds MAX_IMAGE_BYTES. Raises ImageTooLargeError if it can't fit.
    Returns out_path on success.
    """
    page = pdf[page_no - 1]
    scale = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    for _ in range(MAX_DOWNSCALE_PASSES):
        if buf.tell() <= MAX_IMAGE_BYTES:
            break
        w, h = img.size
        img = img.resize((max(1, int(w * 0.75)), max(1, int(h * 0.75))))
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
    if buf.tell() > MAX_IMAGE_BYTES:
        raise ImageTooLargeError(
            f"page {page_no}: {buf.tell()} bytes after "
            f"{MAX_DOWNSCALE_PASSES} downscale passes (cap {MAX_IMAGE_BYTES})"
        )
    out_path.write_bytes(buf.getvalue())
    return out_path
