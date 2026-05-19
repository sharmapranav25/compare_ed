"""Footwear detection + crop helper for the buy-sheet pipeline.

Single-doc only — multi-doc image merging is out of scope. Runs a
serial pass over rendered product pages, detects footwear bboxes with
a YOLO-family detector, crops each shoe to `<doc>.pages/NN.crops/MM.png`,
and writes an `annotated.png` (the page image with numbered red boxes
drawn over each bbox) that the Sonnet matcher in `extract_products.py`
uses to assign crops to SKUs.

Per-page YOLO `imgsz` is chosen by an optional `settings_picker` callback
(see `extract_products.pick_yolo_settings`) — a Sonnet vision call that
classifies the page as low/medium/high density and maps that to imgsz
1280/1920/2560. Catalog pages range from sparse hero shots (Converse
look-book) to 40-shoe grids (adidas catalog); one imgsz can't serve both.

Dependencies (`ultralytics`, torch, PIL) are OPTIONAL imports. When the
package is missing or the weights file is absent, `is_available()` returns
False and the rest of the pipeline runs unchanged — no Image column, no
crops, no errors.

Weights path: `os.environ["DETECT_WEIGHTS"]` if set, else
`<repo>/models/yolov8s-worldv2.pt`. The default is a stock YOLO-World
checkpoint (ultralytics auto-downloads it on first `YOLO(...)` call).
Any fashion-trained YOLO with footwear class names also works — the
module scans `model.names` for keywords (shoe/sneaker/boot/sandal/…)
at load time, and for YOLO-World (open-vocab) it calls `set_classes(...)`
with footwear prompts if no per-class names are found.

Detection runs SERIALLY in a pre-pass invoked from `extract_products.py`,
NOT inside the existing per-page thread pool — `ultralytics.predict()` is
not thread-safe. The matcher Sonnet call happens AFTER detection, in
parallel with the existing extract thread pool, via `process_page`.

Per-page state is persisted in `<NN>.crops/manifest.json`:
  {
    "bboxes": [[x1,y1,x2,y2], ...],
    "crop_paths": ["/abs/.../01.png", ...],
    "imgsz": 1920,
    "imgsz_reasoning": "medium: 10 distinct shoes in a moderate grid"
  }
so re-runs short-circuit detection AND skip the picker call.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import threading
from pathlib import Path
from typing import Callable, Iterable

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


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = REPO_ROOT / "models" / "deepfashion2_yolov8.pt"

FOOTWEAR_KEYWORDS = ("shoe", "sneaker", "boot", "sandal", "heel",
                     "slipper", "footwear", "loafer", "mule", "flat",
                     "pump", "clog")

# Catalog pages routinely show 20-40 small shoe thumbnails on a single
# 300dpi page; the default 640px inference resize squashes them below
# YOLO's detection floor. 1920 keeps small icons recoverable at the
# cost of ~3x inference time per page (still seconds, serial anyway).
DEFAULT_CONF = 0.05
DEFAULT_IMGSZ = 1920
DEFAULT_TARGET_PX = 160
CROP_PAD_FRAC = 0.05  # 5% pad on each side before resize

_MODEL = None
_MODEL_CLASS_IDS: list[int] | None = None
_MODEL_LOCK = threading.Lock()
_LOAD_FAILED = False


def _weights_path() -> Path:
    override = os.environ.get("DETECT_WEIGHTS")
    if override:
        return Path(override).expanduser()
    return DEFAULT_WEIGHTS


def is_available() -> bool:
    """True iff ultralytics+PIL are importable and the weights file exists.

    Doesn't actually load the model — load happens lazily on first detect
    call. A load failure later flips `_LOAD_FAILED` so subsequent calls
    short-circuit without re-trying.
    """
    if _LOAD_FAILED:
        return False
    if not (_HAS_ULTRALYTICS and _HAS_PIL):
        return False
    return _weights_path().exists()


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
        try:
            model = YOLO(str(wpath))
        except Exception as exc:  # noqa: BLE001
            print(f"  detect: failed to load weights at {wpath}: "
                  f"{type(exc).__name__}: {exc} — disabling",
                  file=sys.stderr)
            _LOAD_FAILED = True
            return None
        names = getattr(model, "names", {}) or {}
        ids = _pick_footwear_class_ids(names)
        if hasattr(model, "set_classes") and not ids:
            try:
                model.set_classes(["shoe", "sneaker", "boot", "sandal"])
                names = getattr(model, "names", {}) or {}
                ids = _pick_footwear_class_ids(names) or list(names.keys())
            except Exception:  # noqa: BLE001
                pass
        if not ids:
            print(f"  detect: no footwear classes in {wpath} (names={list(names.values())[:10]}…) — disabling",
                  file=sys.stderr)
            _LOAD_FAILED = True
            return None
        _MODEL = model
        _MODEL_CLASS_IDS = ids
        print(f"  detect: loaded {wpath.name} with footwear classes "
              f"{[names[c] for c in ids]}", file=sys.stderr)
        return _MODEL


def _sort_row_banded(boxes: list[tuple[int, int, int, int]]
                     ) -> list[tuple[int, int, int, int]]:
    """Sort boxes in reading order using row-banding.

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
    anchor_y: float | None = None
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


def detect_shoes(image_path: Path, conf: float = DEFAULT_CONF,
                 imgsz: int = DEFAULT_IMGSZ
                 ) -> list[tuple[int, int, int, int]]:
    """Detect footwear bboxes in pixel coords. Returns reading-order
    sorted list of (x1, y1, x2, y2). Empty list when nothing found or
    when detection isn't available.

    imgsz controls how aggressively the page is downscaled before
    inference. Small (640) is fast but misses tiny grid thumbnails;
    large (1920+) catches small icons but is ~3x slower and over-detects
    on sparse pages. Callers can pass a per-page value picked by a VLM.
    """
    model = get_model()
    if model is None:
        return []
    try:
        results = model.predict(
            source=str(image_path), conf=conf,
            classes=_MODEL_CLASS_IDS, verbose=False,
            agnostic_nms=True, imgsz=imgsz,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  detect: predict failed on {image_path.name}: "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
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


def annotate_page(image_path: Path, bboxes: list[tuple[int, int, int, int]],
                  out_path: Path) -> None:
    """Draw numbered red rectangles over each bbox on the page image and
    save the annotated copy. Used as input to the Sonnet matcher: the VLM
    sees the original page WITH numbered boxes and picks which number
    corresponds to each SKU.

    Numbers start at 1 and follow the order bboxes are passed in (the
    caller is expected to have already sorted them in reading order).
    """
    if Image is None or ImageDraw is None:
        raise RuntimeError("PIL not available")
    with Image.open(image_path) as src:
        annotated = src.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    W, H = annotated.size
    # Stroke width and font size scale with image so labels stay readable
    # on 300dpi catalog pages (~2500px wide) AND smaller renders.
    stroke = max(3, int(max(W, H) * 0.005))
    font_px = max(28, int(max(W, H) * 0.025))
    try:
        font = ImageFont.truetype("Helvetica.ttf", font_px)
    except OSError:
        try:
            font = ImageFont.truetype(
                "/System/Library/Fonts/Helvetica.ttc", font_px)
        except OSError:
            font = ImageFont.load_default()
    for i, (x1, y1, x2, y2) in enumerate(bboxes, start=1):
        draw.rectangle([x1, y1, x2, y2], outline="red", width=stroke)
        label = str(i)
        # Solid red badge in top-left of each box with white number on top.
        badge_w = font_px * (len(label) + 1)
        badge_h = font_px + 8
        bx1, by1 = x1, y1
        bx2, by2 = min(x1 + badge_w, W), min(y1 + badge_h, H)
        draw.rectangle([bx1, by1, bx2, by2], fill="red")
        draw.text((bx1 + font_px // 2, by1 + 2), label,
                  fill="white", font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(out_path, format="PNG", optimize=True)


def crop_and_save(image_path: Path, bbox: tuple[int, int, int, int],
                  out_path: Path, target_px: int = DEFAULT_TARGET_PX) -> None:
    """Crop bbox out of image_path with ~5% pad, longest-side resize to
    target_px, save PNG at out_path."""
    if Image is None:
        raise RuntimeError("PIL not available")
    x1, y1, x2, y2 = bbox
    with Image.open(image_path) as src:
        W, H = src.size
        w, h = x2 - x1, y2 - y1
        pad_x = int(w * CROP_PAD_FRAC)
        pad_y = int(h * CROP_PAD_FRAC)
        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(W, x2 + pad_x)
        cy2 = min(H, y2 + pad_y)
        crop = src.crop((cx1, cy1, cx2, cy2))
        cw, ch = crop.size
        if max(cw, ch) > target_px:
            s = target_px / float(max(cw, ch))
            crop = crop.resize((max(1, int(cw * s)), max(1, int(ch * s))))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(out_path, format="PNG", optimize=True)


SettingsPicker = Callable[[int, Path], tuple[int, dict | None, str | None]]


def run_detect_prepass(pages_to_process: Iterable[tuple[int, Path]],
                       pages_dir: Path, *, force: bool = False,
                       settings_picker: SettingsPicker | None = None,
                       ) -> dict[int, dict]:
    """Serial pre-pass: for each (page_no, page_image_path), detect shoes,
    save crops to `<pages_dir>/NN.crops/MM.png`, write an annotated page
    PNG (`<pages_dir>/NN.crops/annotated.png`) for the Sonnet matcher,
    and persist a `manifest.json`.

    Returns {page_no: {"crops": [paths], "bboxes": [...], "annotated": path}}.
    Pages with zero detections still get a (mostly empty) manifest.

    Resumable: an existing `manifest.json` short-circuits re-detection.
    Pass force=True to clear and redo.

    Skips entirely when `is_available()` is False — returns {}.

    `settings_picker(page_no, image_path) -> (imgsz, usage, reasoning)`
    is called BEFORE detection for each page that needs re-detecting.
    The chosen imgsz is cached in manifest.json so re-runs don't re-pick.
    Pass None to use DEFAULT_IMGSZ for every page.
    """
    out: dict[int, dict] = {}
    if not is_available():
        return out
    if get_model() is None:
        return out
    for page_no, image_path in pages_to_process:
        crops_dir = pages_dir / f"{page_no:02d}.crops"
        manifest_path = crops_dir / "manifest.json"
        annotated_path = crops_dir / "annotated.png"
        if force:
            shutil.rmtree(crops_dir, ignore_errors=True)
        if manifest_path.exists():
            try:
                m = json.loads(manifest_path.read_text())
                cached_bboxes = [tuple(b) for b in m.get("bboxes", [])]
                # Lazily generate the annotated image for old manifests
                # that pre-date the matcher feature.
                if cached_bboxes and not annotated_path.exists():
                    try:
                        annotate_page(image_path, cached_bboxes,
                                      annotated_path)
                    except Exception as exc:  # noqa: BLE001
                        print(f"  detect: lazy annotate failed "
                              f"page={page_no}: {type(exc).__name__}: {exc}",
                              file=sys.stderr)
                out[page_no] = {
                    "crops": list(m.get("crop_paths", [])),
                    "bboxes": cached_bboxes,
                    "annotated": (str(annotated_path)
                                  if annotated_path.exists() else None),
                }
                continue
            except (OSError, json.JSONDecodeError):
                shutil.rmtree(crops_dir, ignore_errors=True)
        imgsz = DEFAULT_IMGSZ
        pick_reasoning: str | None = None
        if settings_picker is not None:
            try:
                imgsz, _usage, pick_reasoning = settings_picker(
                    page_no, image_path,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  detect: settings_picker failed page={page_no}: "
                      f"{type(exc).__name__}: {exc} — using default imgsz",
                      file=sys.stderr)
                imgsz = DEFAULT_IMGSZ
            print(f"  page {page_no:02d}: imgsz={imgsz}"
                  + (f"  ({pick_reasoning})" if pick_reasoning else ""),
                  file=sys.stderr)
        boxes = detect_shoes(image_path, imgsz=imgsz)
        crops_dir.mkdir(parents=True, exist_ok=True)
        crop_paths: list[str] = []
        for i, box in enumerate(boxes, start=1):
            cp = crops_dir / f"{i:02d}.png"
            try:
                crop_and_save(image_path, box, cp)
            except Exception as exc:  # noqa: BLE001
                print(f"  detect: crop failed page={page_no} box={box}: "
                      f"{type(exc).__name__}: {exc}", file=sys.stderr)
                continue
            crop_paths.append(str(cp.resolve()))
        if boxes:
            try:
                annotate_page(image_path, boxes, annotated_path)
            except Exception as exc:  # noqa: BLE001
                print(f"  detect: annotate failed page={page_no}: "
                      f"{type(exc).__name__}: {exc}", file=sys.stderr)
        manifest_path.write_text(json.dumps(
            {"bboxes": [list(b) for b in boxes],
             "crop_paths": crop_paths,
             "imgsz": imgsz,
             "imgsz_reasoning": pick_reasoning},
            indent=2,
        ))
        out[page_no] = {
            "crops": crop_paths,
            "bboxes": [tuple(b) for b in boxes],
            "annotated": (str(annotated_path)
                          if annotated_path.exists() else None),
        }
        if boxes:
            print(f"  page {page_no:02d}: detected {len(boxes)} footwear "
                  f"crops -> {crops_dir.name}/", file=sys.stderr)
    return out
