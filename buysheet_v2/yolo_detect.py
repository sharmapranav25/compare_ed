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


def detect_shoes_in_image(
    png_bytes: bytes, *, conf: float = DEFAULT_CONF, imgsz: int = DEFAULT_IMGSZ,
) -> list[tuple[int, int, int, int]]:
    """Detect footwear bboxes in a page render (bytes form). Returns
    reading-order sorted list of (x1, y1, x2, y2) in pixel coords of the
    input image. Empty list when nothing found or detection unavailable.

    imgsz controls how aggressively the page is downscaled before
    inference. 1280=fast/missesTiny, 1920=balanced, 2560=slow/catchesTiny.
    Pranav's pick_yolo_density() picks per-page via a Sonnet density probe.
    """
    model = get_model()
    if model is None:
        return []
    # ultralytics predict accepts PIL.Image directly — saves writing the
    # PNG to disk just to read it back.
    if Image is None:
        return []
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    try:
        results = model.predict(
            source=img, conf=conf, classes=_MODEL_CLASS_IDS,
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


def annotate_page(
    png_bytes: bytes, bboxes: list[tuple[int, int, int, int]], out_path: Path,
) -> None:
    """Draw numbered red rectangles over each bbox on the page image and
    save the annotated copy. Used as input to the Sonnet matcher: the VLM
    sees the page WITH numbered boxes and picks which number corresponds
    to each SKU.

    Numbers start at 1 and follow the order bboxes are passed in (caller
    is expected to have sorted them in reading order via _sort_row_banded).
    """
    if Image is None or ImageDraw is None:
        raise RuntimeError("PIL not available")
    src = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    annotated = src.copy()
    draw = ImageDraw.Draw(annotated)
    W, H = annotated.size
    # Stroke width and font size scale with image so labels stay readable
    # at both 300dpi catalog renders AND smaller VLM-target renders.
    stroke = max(3, int(max(W, H) * 0.005))
    font_px = max(20, int(max(W, H) * 0.025))
    try:
        font = ImageFont.truetype("Helvetica.ttf", font_px)
    except OSError:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_px)
        except OSError:
            font = ImageFont.load_default()
    for i, (x1, y1, x2, y2) in enumerate(bboxes, start=1):
        draw.rectangle([x1, y1, x2, y2], outline="red", width=stroke)
        label = str(i)
        badge_w = font_px * (len(label) + 1)
        badge_h = font_px + 8
        bx2 = min(x1 + badge_w, W)
        by2 = min(y1 + badge_h, H)
        draw.rectangle([x1, y1, bx2, by2], fill="red")
        draw.text((x1 + font_px // 2, y1 + 2), label, fill="white", font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(out_path, format="PNG", optimize=True)


# Per-page settings picker: (page) -> (imgsz, reasoning)
# Wraps photo_match.pick_yolo_density to keep yolo_detect.py importable
# even when ANTHROPIC_API_KEY isn't set (e.g. tests).
DensityPicker = Callable[[IngestedPage], tuple[int, Optional[str]]]


def run_detect_prepass(
    pages: Iterable[IngestedPage], sidecar_dir: Path, *,
    pdf_stem: str, force: bool = False,
    density_picker: Optional[DensityPicker] = None,
) -> dict[int, dict]:
    """Serial pre-pass over all pages. For each: pick imgsz, run YOLO,
    annotate, persist. Returns {page_no: {bboxes, imgsz, imgsz_reasoning,
    annotated_path}}.

    Sidecar layout (mirrors plan's caching pattern):
      <sidecar_dir>/<pdf_stem>.v2.yolo.json
      <sidecar_dir>/<pdf_stem>.v2.yolo/page_NN_annotated.png  (per page)

    Resumable: if the sidecar JSON exists AND every annotated.png exists,
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

    pages_list = list(pages)
    log.info("yolo_detect: %d pages to scan (cached=%d)", len(pages_list), len(cached))
    for page in pages_list:
        page_no = page.page_no
        annotated_path = annotated_dir / f"page_{page_no:03d}_annotated.png"
        entry = cached.get(page_no)
        if entry and annotated_path.exists() and not force:
            out[page_no] = {
                "bboxes": [tuple(b) for b in entry.get("bboxes", [])],
                "imgsz": entry.get("imgsz", DEFAULT_IMGSZ),
                "imgsz_reasoning": entry.get("imgsz_reasoning"),
                "annotated_path": str(annotated_path),
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

        bboxes = detect_shoes_in_image(page.png_bytes, imgsz=imgsz)
        if bboxes:
            try:
                annotate_page(page.png_bytes, bboxes, annotated_path)
            except Exception as exc:  # noqa: BLE001
                log.warning("annotate failed on page %d: %s: %s",
                            page_no, type(exc).__name__, exc)
        log.info("  page %d: imgsz=%d detected=%d (%s)",
                 page_no, imgsz, len(bboxes), reasoning or "default")
        out[page_no] = {
            "bboxes": bboxes,
            "imgsz": imgsz,
            "imgsz_reasoning": reasoning,
            "annotated_path": str(annotated_path) if annotated_path.exists() else None,
        }

    # Persist sidecar with all entries (cached + new)
    serializable = {
        str(p): {
            "bboxes": [list(b) for b in v["bboxes"]],
            "imgsz": v["imgsz"],
            "imgsz_reasoning": v["imgsz_reasoning"],
            "annotated_path": v["annotated_path"],
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
