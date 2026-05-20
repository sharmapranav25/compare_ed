"""YOLO-based shoe detection pre-pass.

For each rendered catalog page, run YOLO-World (open-vocabulary detector
constrained to footwear classes) → list of bboxes → annotated page render
with numbered red boxes → cache to a per-PDF sidecar so re-runs are free.
The annotated PNG is fed into the Sonnet matcher (photo_match.py) which
assigns each extracted SKU to one of the numbered boxes.

This decouples "where are shoes?" (YOLO, layout-agnostic, visual pattern)
from "which SKU is which?" (Sonnet, visual reasoning over numbered boxes)
— much more robust than the text-anchored photo_vlm strategy that breaks
on catalogs where photos aren't reliably near SKU text (Nike PPTX,
Mizuno lookbook).

Adapted from Pranav's `detect/shoe.py` in
~/Desktop/untitled folder 2/compare_ed/. Key differences from his version:
  - Operates on our IngestedPage objects (in-memory PNG bytes from
    lifted/pdf_render.py), not on saved page PNG files. Coords are
    therefore in our 1568-px-long-edge render space.
  - Single sidecar `<pdf_stem>.v2.yolo.json` per PDF instead of per-page
    `<NN>.crops/manifest.json` directories.
  - Annotated PNGs land in `<pdf_stem>.v2.yolo/page_NN_annotated.png`.
  - No per-crop disk save (write.py/sheets_writer.py crop on-the-fly
    from the page render using the bbox; matches our existing flow).
  - is_available() / silent-no-op pattern preserved from Pranav so the
    pipeline degrades cleanly if ultralytics/weights are missing.

Dependencies (`ultralytics`, PIL) are OPTIONAL imports. When the package
is missing or the weights file is absent, `is_available()` returns False
and the pipeline falls back to the legacy photo_vlm path.

Weights path: `os.environ["DETECT_WEIGHTS"]` if set, else
`buysheet_v2/models/yolov8s-worldv2.pt`. The default is the stock
YOLO-World checkpoint (ultralytics auto-downloads it on first
`YOLO(...)` call, ~25 MB).

Detection runs SERIALLY in a pre-pass invoked from pipeline.py BEFORE
the per-page extract step — `ultralytics.predict()` is not thread-safe.
"""
from __future__ import annotations

import io
import json
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Callable, Iterable, Optional

try:
    from ultralytics import YOLO  # type: ignore
    _HAS_ULTRALYTICS = True
except ImportError:
    YOLO = None  # type: ignore
    _HAS_ULTRALYTICS = False

try:
    from PIL import Image, ImageDraw, ImageFont  # type: ignore
    _HAS_PIL = True
except ImportError:
    Image = None  # type: ignore
    ImageDraw = None  # type: ignore
    ImageFont = None  # type: ignore
    _HAS_PIL = False

try:
    import pypdfium2 as pdfium  # type: ignore
except ImportError:
    pdfium = None  # type: ignore

from buysheet_v2.ingest import IngestedPage

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = REPO_ROOT / "models" / "yolov8s-worldv2.pt"

# Vocabulary that YOLO-World's open-vocab head is told to detect. We start
# narrow (just shoe-likes); broader prompts produce more false positives
# (e.g. "footwear" matches socks/insoles too).
FOOTWEAR_PROMPTS = ["shoe", "sneaker", "boot", "sandal"]
FOOTWEAR_KEYWORDS = ("shoe", "sneaker", "boot", "sandal", "heel",
                     "slipper", "footwear", "loafer", "mule", "flat",
                     "pump", "clog")

# Catalog pages routinely show 20-40 small shoe thumbnails on a single
# 300dpi page; the default 640px inference resize squashes them below
# YOLO's detection floor. 1920 keeps small icons recoverable at the cost
# of ~3x inference time per page (still seconds, serial anyway).
DEFAULT_CONF = 0.05
DEFAULT_IMGSZ = 1920

_MODEL = None
_MODEL_CLASS_IDS: Optional[list[int]] = None
_MODEL_LOCK = threading.Lock()
_LOAD_FAILED = False


def _weights_path() -> Path:
    override = os.environ.get("DETECT_WEIGHTS")
    if override:
        return Path(override).expanduser()
    return DEFAULT_WEIGHTS


def is_available() -> bool:
    """True iff ultralytics + PIL are importable. Weights file may or may
    not exist yet — ultralytics auto-downloads on first YOLO() call.

    A load failure later flips `_LOAD_FAILED` so subsequent calls
    short-circuit without re-trying for the rest of the process lifetime.
    """
    if _LOAD_FAILED:
        return False
    return _HAS_ULTRALYTICS and _HAS_PIL


def _pick_footwear_class_ids(names: dict) -> list[int]:
    """Return class indices whose name matches a footwear keyword."""
    out: list[int] = []
    for cid, name in names.items():
        if any(kw in str(name).lower() for kw in FOOTWEAR_KEYWORDS):
            out.append(int(cid))
    return out


def get_model():
    """Lazy singleton. Returns the loaded YOLO model, or None if load failed.

    Thread-safe construction (lock held only during the load). Inference
    itself is NOT thread-safe — callers must serialize predict() calls.

    On first call: ultralytics may need to download the weights file from
    its CDN if the weights file isn't already in models/. ~25 MB download.
    """
    global _MODEL, _MODEL_CLASS_IDS, _LOAD_FAILED
    if _MODEL is not None:
        return _MODEL
    if _LOAD_FAILED or not is_available():
        return None
    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL
        if _LOAD_FAILED:
            return None
        wpath = _weights_path()
        # ultralytics resolves the file: if absolute, uses as-is; if just a
        # name, looks in CWD and auto-downloads from their CDN if missing.
        # Pass the filename only so the CDN fallback works even when our
        # default path doesn't exist yet.
        try:
            model = YOLO(str(wpath) if wpath.exists() else wpath.name)
        except Exception as exc:  # noqa: BLE001
            log.warning("yolo_detect: failed to load weights at %s: %s: %s — disabling",
                        wpath, type(exc).__name__, exc)
            _LOAD_FAILED = True
            return None
        names = getattr(model, "names", {}) or {}
        ids = _pick_footwear_class_ids(names)
        if hasattr(model, "set_classes") and not ids:
            # YOLO-World (open-vocab): set the prompt vocabulary
            try:
                model.set_classes(FOOTWEAR_PROMPTS)
                names = getattr(model, "names", {}) or {}
                ids = _pick_footwear_class_ids(names) or list(names.keys())
            except Exception:  # noqa: BLE001
                pass
        if not ids:
            log.warning("yolo_detect: no footwear classes in %s (names=%s…) — disabling",
                        wpath, list(names.values())[:10])
            _LOAD_FAILED = True
            return None
        _MODEL = model
        _MODEL_CLASS_IDS = ids
        log.info("yolo_detect: loaded %s with footwear classes %s",
                 wpath.name, [names[c] for c in ids])
        return _MODEL


def _sort_row_banded(boxes: list[tuple[int, int, int, int]]
                     ) -> list[tuple[int, int, int, int]]:
    """Sort boxes in reading order (top-to-bottom, left-to-right) using
    row-banding so boxes on the same visual row stay grouped.

    Groups boxes whose Y-centers fall within `band = 0.6 * mean(height)`
    of a band's anchor Y. Within each band, sort by X-center. Concatenate
    bands top→bottom. Handles 2x3 grid look-book layouts where rows are
    not perfectly y-aligned.
    """
    if not boxes:
        return []
    heights = [(y2 - y1) for x1, y1, x2, y2 in boxes]
    band = max(1.0, 0.6 * (sum(heights) / len(heights)))
    centers = [((x1 + x2) / 2.0, (y1 + y2) / 2.0, box)
               for box in boxes for x1, y1, x2, y2 in [box]]
    centers.sort(key=lambda t: t[1])  # by y
    bands: list[list[tuple[float, float, tuple]]] = []
    anchor_y: Optional[float] = None
    for cx, cy, box in centers:
        if anchor_y is None or cy - anchor_y > band:
            bands.append([])
            anchor_y = cy
        bands[-1].append((cx, cy, box))
    out: list[tuple[int, int, int, int]] = []
    for b in bands:
        b.sort(key=lambda t: t[0])
        out.extend(box for _, _, box in b)
    return out


def detect_shoes_in_pil(
    pil_img: "Image.Image", *, conf: float = DEFAULT_CONF,
    imgsz: int = DEFAULT_IMGSZ,
) -> list[tuple[int, int, int, int]]:
    """Detect footwear bboxes on a PIL image directly. Returns reading-order
    sorted list of (x1, y1, x2, y2) in the **input image's pixel coords**.

    Callers control the source resolution: passing a 1568-px IngestedPage
    makes density tiers (1280/1920/2560) useless on medium+high because
    YOLO will upscale (no new info). Passing a 3000-px hi-res render lets
    high=2560 actually recover detail on dense grids. See run_detect_prepass
    for the hi-res routing decision.
    """
    model = get_model()
    if model is None:
        return []
    if Image is None:
        return []
    try:
        results = model.predict(
            source=pil_img, conf=conf, classes=_MODEL_CLASS_IDS,
            verbose=False, agnostic_nms=True, imgsz=imgsz,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("yolo_detect: predict failed: %s: %s",
                    type(exc).__name__, exc)
        return []
    boxes: list[tuple[int, int, int, int]] = []
    for r in results:
        b = getattr(r, "boxes", None)
        if b is None:
            continue
        xyxy = getattr(b, "xyxy", None)
        if xyxy is None:
            continue
        for row in xyxy.tolist():
            x1, y1, x2, y2 = (int(round(v)) for v in row[:4])
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
    return _sort_row_banded(boxes)


def detect_shoes_in_image(
    png_bytes: bytes, *, conf: float = DEFAULT_CONF, imgsz: int = DEFAULT_IMGSZ,
) -> list[tuple[int, int, int, int]]:
    """Thin wrapper: PNG bytes → PIL → detect_shoes_in_pil. Returns bboxes
    in the source image's coord space (typically 1568-px lores)."""
    if Image is None:
        return []
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    return detect_shoes_in_pil(img, conf=conf, imgsz=imgsz)


def _load_font(font_px: int):
    """Pick a readable font with graceful fallbacks (Helvetica → /System →
    PIL default). Parent's annotate uses 28-px minimum on the hi-res canvas;
    we keep 20-px for the legacy lo-res annotate, bump to 28 for hi-res.
    """
    try:
        return ImageFont.truetype("Helvetica.ttf", font_px)
    except OSError:
        try:
            return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc",
                                     font_px)
        except OSError:
            return ImageFont.load_default()


def _draw_numbered_bboxes(annotated, bboxes, stroke: int, font_px: int) -> None:
    """Shared rectangle+badge drawing used by both lo-res and hi-res paths."""
    draw = ImageDraw.Draw(annotated)
    W, H = annotated.size
    font = _load_font(font_px)
    for i, (x1, y1, x2, y2) in enumerate(bboxes, start=1):
        draw.rectangle([x1, y1, x2, y2], outline="red", width=stroke)
        label = str(i)
        badge_w = font_px * (len(label) + 1)
        badge_h = font_px + 8
        bx2 = min(x1 + badge_w, W)
        by2 = min(y1 + badge_h, H)
        draw.rectangle([x1, y1, bx2, by2], fill="red")
        draw.text((x1 + font_px // 2, y1 + 2), label, fill="white", font=font)


def annotate_page(
    png_bytes: bytes, bboxes: list[tuple[int, int, int, int]], out_path: Path,
) -> None:
    """Lo-res annotation path (LEGACY fallback).

    Draws numbered red rectangles over each bbox on the 1568-px page image
    and saves the annotated copy as PNG. Used when the hi-res path
    (annotate_page_at_scale) is unavailable — pdf doc not passed in, or
    ImageTooLargeError fired even after the encode cascade.
    """
    if Image is None or ImageDraw is None:
        raise RuntimeError("PIL not available")
    src = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    annotated = src.copy()
    W, H = annotated.size
    stroke = max(3, int(max(W, H) * 0.005))
    font_px = max(20, int(max(W, H) * 0.025))
    _draw_numbered_bboxes(annotated, bboxes, stroke, font_px)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(out_path, format="PNG", optimize=True)


def _annotate_image_and_save(
    annotated: "Image.Image",
    bboxes_in_image_space: list[tuple[int, int, int, int]],
    out_base_path: Path,
) -> Path:
    """Core: draw numbered red rectangles on the given PIL image + save via
    `save_with_cascade` (PNG → JPEG-q85 fallback). Caller is responsible
    for ensuring `bboxes_in_image_space` matches `annotated`'s pixel coords.

    Returns the actual saved path (suffix may be .png or .jpg). Raises
    `ImageTooLargeError` from the cascade if no encode strategy fits.
    """
    if Image is None or ImageDraw is None:
        raise RuntimeError("PIL not available")
    from buysheet_v2.lifted.pdf_render import save_with_cascade
    W, H = annotated.size
    stroke = max(3, int(max(W, H) * 0.005))
    # Parent repo uses 28-px font min on the hi-res canvas; match that.
    font_px = max(28, int(max(W, H) * 0.025))
    _draw_numbered_bboxes(annotated, bboxes_in_image_space, stroke, font_px)
    return save_with_cascade(annotated, out_base_path)


def annotate_page_at_scale(
    pdf: "pdfium.PdfDocument", page_no_1: int,
    bboxes_lores: list[tuple[int, int, int, int]],
    lores_w: int, lores_h: int,
    out_base_path: Path,
    target_long_edge: int = None,
) -> Path:
    """Hi-res annotation entry point that ALSO renders the page.

    Used when caller has only lo-res bboxes and needs to produce a hi-res
    annotated PNG. Renders at `target_long_edge` (default 3000), scales
    bboxes 1568→hires, draws + saves. Hi-res coords confined to this call.

    Prefer the inline path in `run_detect_prepass` (which calls
    `_annotate_image_and_save` directly on a hi-res image it already has)
    so the page isn't rendered twice — once for YOLO inference, once for
    annotation.
    """
    if Image is None or ImageDraw is None:
        raise RuntimeError("PIL not available")
    from buysheet_v2.lifted.pdf_render import (
        MATCHER_MAX_LONG_EDGE_PX, render_page_for_matcher,
    )
    if target_long_edge is None:
        target_long_edge = MATCHER_MAX_LONG_EDGE_PX
    annotated, hi_w, hi_h = render_page_for_matcher(
        pdf, page_no_1, max_long_edge=target_long_edge,
    )
    lores_long = max(lores_w, lores_h)
    hires_long = max(hi_w, hi_h)
    scale = hires_long / float(lores_long) if lores_long > 0 else 1.0
    bboxes_hires = [
        (int(round(x1 * scale)), int(round(y1 * scale)),
         int(round(x2 * scale)), int(round(y2 * scale)))
        for (x1, y1, x2, y2) in bboxes_lores
    ]
    return _annotate_image_and_save(annotated, bboxes_hires, out_base_path)


# Per-page settings picker: (page) -> (imgsz, reasoning)
# Wraps photo_match.pick_yolo_density to keep yolo_detect.py importable
# even when ANTHROPIC_API_KEY isn't set (e.g. tests).
DensityPicker = Callable[[IngestedPage], tuple[int, Optional[str]]]


def run_detect_prepass(
    pages: Iterable[IngestedPage], sidecar_dir: Path, *,
    pdf_stem: str, force: bool = False,
    density_picker: Optional[DensityPicker] = None,
    pdf: "Optional[pdfium.PdfDocument]" = None,
) -> dict[int, dict]:
    """Serial pre-pass over all pages. For each: pick imgsz, run YOLO,
    annotate, persist. Returns {page_no: {bboxes, imgsz, imgsz_reasoning,
    annotated_path, matcher_low_res?}}.

    Sidecar layout:
      <sidecar_dir>/<pdf_stem>.v2.yolo.json
      <sidecar_dir>/<pdf_stem>.v2.yolo/page_NN_annotated.{png,jpg}

    Hi-res annotation (3000px long-edge) is used when `pdf` is supplied AND
    the encode cascade fits the result under Anthropic's image cap. The
    annotated file's extension reflects which encoder strategy won — PNG
    when text-heavy, JPEG-q85 when photo-heavy. Bboxes in the sidecar
    remain in 1568-px space regardless.

    Fallback path: if hi-res render raises ImageTooLargeError (no strategy
    fits), or if `pdf` is None, falls back to annotating the lo-res
    (1568-px) IngestedPage.png_bytes. That page's sidecar entry gets
    `matcher_low_res=True` so write.py / the buyer can see when the
    matcher saw a degraded image.

    Resumable: if the sidecar JSON exists AND every annotated image exists,
    skip YOLO entirely. Pass force=True to redo from scratch.

    Returns {} if YOLO is unavailable. Caller is expected to fall back to
    photo_vlm in that case.
    """
    out: dict[int, dict] = {}
    if not is_available():
        log.info("yolo_detect: not available (ultralytics or PIL missing) — skipping")
        return out
    if get_model() is None:
        log.warning("yolo_detect: model load failed — skipping")
        return out

    # Lazy import keeps tests/imports cheap when lifted/* isn't usable.
    try:
        from buysheet_v2.lifted.pdf_render import ImageTooLargeError
    except ImportError:
        ImageTooLargeError = RuntimeError  # type: ignore

    sidecar_dir.mkdir(parents=True, exist_ok=True)
    sidecar_json = sidecar_dir / f"{pdf_stem}.v2.yolo.json"
    annotated_dir = sidecar_dir / f"{pdf_stem}.v2.yolo"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    cached: dict[int, dict] = {}
    if sidecar_json.exists() and not force:
        try:
            raw = json.loads(sidecar_json.read_text())
            cached = {int(k): v for k, v in raw.get("detections", {}).items()}
        except (json.JSONDecodeError, OSError, ValueError):
            cached = {}

    def _existing_annotated(base: Path) -> Optional[Path]:
        """Match either .png (legacy + PNG-cascade winners) or .jpg (JPEG)."""
        for ext in (".png", ".jpg"):
            p = base.with_suffix(ext)
            if p.exists():
                return p
        return None

    pages_list = list(pages)
    log.info("yolo_detect: %d pages to scan (cached=%d)", len(pages_list), len(cached))
    for page in pages_list:
        page_no = page.page_no
        annotated_base = annotated_dir / f"page_{page_no:03d}_annotated"
        cached_annotated = _existing_annotated(annotated_base)
        entry = cached.get(page_no)
        if entry and cached_annotated is not None and not force:
            out[page_no] = {
                "bboxes": [tuple(b) for b in entry.get("bboxes", [])],
                "imgsz": entry.get("imgsz", DEFAULT_IMGSZ),
                "imgsz_reasoning": entry.get("imgsz_reasoning"),
                "annotated_path": str(cached_annotated),
                "matcher_low_res": entry.get("matcher_low_res", False),
            }
            continue

        imgsz = DEFAULT_IMGSZ
        reasoning: Optional[str] = None
        if density_picker is not None:
            try:
                imgsz, reasoning = density_picker(page)
            except Exception as exc:  # noqa: BLE001
                log.warning("density picker failed on page %d: %s: %s — using default imgsz",
                            page_no, type(exc).__name__, exc)
                imgsz = DEFAULT_IMGSZ

        bboxes: list[tuple[int, int, int, int]] = []
        annotated_path: Optional[Path] = None
        matcher_low_res = False
        hires_yolo = False  # True iff YOLO actually ran on the hi-res render

        if pdf is not None:
            # Hi-res inference + annotation share a single render. The
            # density picker's tiers (1280/1920/2560) only recover detail
            # when the source actually has it — running YOLO on the 1568-px
            # IngestedPage caps inference quality regardless of imgsz. The
            # 3000-px render lets imgsz=2560 downscale-with-detail instead
            # of upscale-from-nothing.
            try:
                from buysheet_v2.lifted.pdf_render import (
                    MATCHER_MAX_LONG_EDGE_PX, render_page_for_matcher,
                )
                hires_img, hi_w, hi_h = render_page_for_matcher(
                    pdf, page_no, max_long_edge=MATCHER_MAX_LONG_EDGE_PX,
                )
                bboxes_hires = detect_shoes_in_pil(hires_img, imgsz=imgsz)
                # Scale hi-res bboxes back to 1568-px space for the sidecar
                # so downstream consumers (write._crop_card_png, photo_match
                # lookup) keep operating in the canonical coordinate frame.
                lores_long = max(page.width_px, page.height_px)
                hires_long = max(hi_w, hi_h)
                scale_down = (lores_long / float(hires_long)
                              if hires_long > 0 else 1.0)
                bboxes = [
                    (int(round(x1 * scale_down)), int(round(y1 * scale_down)),
                     int(round(x2 * scale_down)), int(round(y2 * scale_down)))
                    for (x1, y1, x2, y2) in bboxes_hires
                ]
                hires_yolo = True
                # Annotate using the SAME hi-res image we ran YOLO on, with
                # the hi-res bboxes — no second render, no extra scaling.
                if bboxes_hires:
                    try:
                        annotated_path = _annotate_image_and_save(
                            hires_img, bboxes_hires, annotated_base,
                        )
                    except ImageTooLargeError as exc:
                        log.warning(
                            "hi-res annotate exceeded byte cap on page %d: "
                            "%s — falling back to 1568-px annotation",
                            page_no, exc,
                        )
                        matcher_low_res = True
                    except Exception as exc:  # noqa: BLE001
                        log.warning(
                            "hi-res annotate failed on page %d: %s: %s — "
                            "falling back to lo-res", page_no,
                            type(exc).__name__, exc,
                        )
                        matcher_low_res = True
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "hi-res YOLO path failed on page %d: %s: %s — falling "
                    "back to lo-res inference", page_no,
                    type(exc).__name__, exc,
                )
                hires_yolo = False
                matcher_low_res = True

        if not hires_yolo:
            # Lo-res path: pdf wasn't supplied or hi-res path errored. YOLO
            # inferences on the 1568-px IngestedPage; density tiers will
            # over-upscale on medium/high.
            bboxes = detect_shoes_in_image(page.png_bytes, imgsz=imgsz)
            matcher_low_res = True

        if bboxes and annotated_path is None:
            # Lo-res annotation fallback (always .png).
            fallback_path = annotated_base.with_suffix(".png")
            try:
                annotate_page(page.png_bytes, bboxes, fallback_path)
                annotated_path = fallback_path
            except Exception as exc:  # noqa: BLE001
                log.warning("annotate failed on page %d: %s: %s",
                            page_no, type(exc).__name__, exc)
        log.info(
            "  page %d: imgsz=%d detected=%d (%s)%s",
            page_no, imgsz, len(bboxes), reasoning or "default",
            " [matcher_low_res]" if matcher_low_res else "",
        )
        out[page_no] = {
            "bboxes": bboxes,
            "imgsz": imgsz,
            "imgsz_reasoning": reasoning,
            "annotated_path": str(annotated_path) if annotated_path else None,
            "matcher_low_res": matcher_low_res,
        }

    # Persist sidecar with all entries (cached + new)
    serializable = {
        str(p): {
            "bboxes": [list(b) for b in v["bboxes"]],
            "imgsz": v["imgsz"],
            "imgsz_reasoning": v["imgsz_reasoning"],
            "annotated_path": v["annotated_path"],
            "matcher_low_res": v.get("matcher_low_res", False),
        }
        for p, v in out.items()
    }
    sidecar_json.write_text(json.dumps({
        "weights_path": str(_weights_path()),
        "vocab": FOOTWEAR_PROMPTS,
        "detections": serializable,
    }, indent=2))
    log.info("yolo_detect: sidecar written to %s", sidecar_json.name)
    return out
