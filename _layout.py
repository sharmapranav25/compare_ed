"""DocLayout-YOLO layout detector + per-vendor calibration helpers.

Why this file exists:
  Buy-sheet pages need a heuristic photo-locator that works on rasterized
  pages (where PyMuPDF's image XObject enumeration finds nothing). YOLO
  trained on document layout (DocStructBench) gives us figure / text /
  title boxes at the pixel level on the full-res rendered PNG.

  The model and weights are heavy enough that we lazy-load on first use,
  share a single instance across threads, and cache the .pt file under
  ~/.cache/doclayout-yolo/.

Box format throughout:
  {"cls": "figure" | "plain text" | "title" | ...,
   "bbox": [x1, y1, x2, y2],   # pixels in the full-res PNG
   "conf": 0.92}

Public surface:
  detect_yolo(png_path)                -> list[Box]                 — raster detection
  calibrate_per_vendor(yolo_pages,     -> {vendor_key: stats_dict}  — size/aspect priors
                       page_records)
  YoloUnavailable                                                    — sentinel exception

If `doclayout-yolo` (and its torch backbone) isn't installed, the helpers
raise YoloUnavailable on first call so callers can degrade gracefully.
"""

from __future__ import annotations

import math
import os
import sys
import threading
from pathlib import Path
from statistics import median

WEIGHTS_REPO_ID = "juliozhao/DocLayout-YOLO-DocStructBench"
WEIGHTS_FILENAME = "doclayout_yolo_docstructbench_imgsz1024.pt"
DEFAULT_IMGSZ = 1024
DEFAULT_CONF = 0.2

CACHE_DIR = Path(os.path.expanduser("~/.cache/doclayout-yolo"))


class YoloUnavailable(RuntimeError):
    """Raised when doclayout-yolo (or its torch backbone) isn't installed."""


_model = None
_model_lock = threading.Lock()


def _device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


def _ensure_weights() -> Path:
    """Resolve the .pt path, downloading via huggingface_hub if needed.
    Returns the local file path; raises YoloUnavailable if HF download
    machinery isn't importable."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    local = CACHE_DIR / WEIGHTS_FILENAME
    if local.exists() and local.stat().st_size > 0:
        return local
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise YoloUnavailable(
            "huggingface_hub is required to download DocLayout-YOLO weights"
        ) from exc
    fetched = hf_hub_download(
        repo_id=WEIGHTS_REPO_ID,
        filename=WEIGHTS_FILENAME,
        cache_dir=str(CACHE_DIR),
    )
    return Path(fetched)


def _load_model():
    """Lazy singleton model load. Thread-safe."""
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        try:
            from doclayout_yolo import YOLOv10
        except ImportError as exc:
            raise YoloUnavailable(
                "doclayout-yolo is not installed (pip install doclayout-yolo)"
            ) from exc
        weights = _ensure_weights()
        _model = YOLOv10(str(weights))
        print(f"[layout] loaded DocLayout-YOLO weights from {weights}",
              file=sys.stderr)
        return _model


def detect_yolo(png_path: Path, imgsz: int = DEFAULT_IMGSZ,
                conf: float = DEFAULT_CONF) -> list[dict]:
    """Run DocLayout-YOLO on a single PNG. Returns a list of Box dicts.

    Bbox coordinates are in pixel space of the input PNG (NOT resized to
    imgsz — YOLO handles the internal resize and reports back in original
    pixels). conf threshold is intentionally low (0.2) — we'd rather
    over-collect candidates and filter via size/aspect downstream.
    """
    model = _load_model()
    # doclayout-yolo's predict mirrors ultralytics: returns a list of Results
    # objects, one per input image. Each has .boxes with xyxy + cls + conf.
    res = model.predict(str(png_path), imgsz=imgsz, conf=conf,
                        device=_device(), verbose=False)
    if not res:
        return []
    r = res[0]
    names = getattr(r, "names", None) or getattr(model, "names", {}) or {}
    boxes_out: list[dict] = []
    boxes = getattr(r, "boxes", None)
    if boxes is None:
        return []
    # Convert to lists once — avoids touching torch tensors deeper in
    xyxy = boxes.xyxy.cpu().numpy().tolist() if hasattr(boxes.xyxy, "cpu") else list(boxes.xyxy)
    cls = boxes.cls.cpu().numpy().tolist() if hasattr(boxes.cls, "cpu") else list(boxes.cls)
    confs = boxes.conf.cpu().numpy().tolist() if hasattr(boxes.conf, "cpu") else list(boxes.conf)
    for bb, c, cf in zip(xyxy, cls, confs):
        x1, y1, x2, y2 = [float(v) for v in bb]
        cls_id = int(c)
        cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else (
            names[cls_id] if 0 <= cls_id < len(names) else str(cls_id)
        )
        boxes_out.append({
            "cls": str(cls_name),
            "bbox": [x1, y1, x2, y2],
            "conf": float(cf),
        })
    return boxes_out


# ---- Calibration ----------------------------------------------------------


def _is_figure_class(cls: str) -> bool:
    c = cls.lower()
    return "figure" in c or c == "image"


def _is_text_class(cls: str) -> bool:
    c = cls.lower()
    if _is_figure_class(c):
        return False
    return any(t in c for t in ("text", "title", "caption", "list",
                                "abandon", "footnote", "header", "footer"))


def figure_boxes(boxes: list[dict]) -> list[dict]:
    return [b for b in boxes if _is_figure_class(b["cls"])]


def text_boxes(boxes: list[dict]) -> list[dict]:
    return [b for b in boxes if _is_text_class(b["cls"])]


def _bbox_wh(bbox: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1), max(0.0, y2 - y1)


def _iqr(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    vs = sorted(values)
    n = len(vs)
    q1 = vs[max(0, n // 4)]
    q3 = vs[min(n - 1, (3 * n) // 4)]
    return max(0.0, q3 - q1)


def _stats(widths: list[float], heights: list[float]) -> dict:
    if not widths:
        return {
            "median_w": None, "median_h": None, "median_aspect": None,
            "iqr_w": 0.0, "iqr_h": 0.0, "iqr_aspect": 0.0,
            "n_boxes": 0,
        }
    aspects = [w / h for w, h in zip(widths, heights) if h > 0]
    return {
        "median_w": float(median(widths)),
        "median_h": float(median(heights)),
        "median_aspect": float(median(aspects)) if aspects else None,
        "iqr_w": float(_iqr(widths)),
        "iqr_h": float(_iqr(heights)),
        "iqr_aspect": float(_iqr(aspects)) if aspects else 0.0,
        "n_boxes": len(widths),
    }


PYMUPDF_MIN_SIZE_PX = 60  # rects smaller than this on either axis are
                          # almost certainly swatch dots / icons /
                          # chrome graphics, not product photos. Excluded
                          # from the calibration sample so they don't drag
                          # the median below the real-photo size band.


def calibrate_pymupdf_per_vendor(pymupdf_pages: list[dict],
                                 page_records: list[dict],
                                 xref_pages: dict[int, int] | None = None,
                                 recurring_threshold: int | None = None,
                                 min_size_px: float = PYMUPDF_MIN_SIZE_PX
                                 ) -> dict:
    """Size/aspect priors for PyMuPDF image-XObject rects grouped by vendor.

    PyMuPDF rects are exact PDF objects — they should NOT be filtered by
    the YOLO-derived size band (YOLO collapses dense grids into a single
    huge figure on some layouts, which makes the YOLO median useless for
    sizing real photos). Hence this independent calibration.

    Calibration sample exclusions:
      - rects with xref appearing on >= `recurring_threshold` pages
        (logos, banners — uniform size across pages, skews median)
      - rects below `min_size_px` on either axis
        (color-swatch dots, tiny icons, chrome graphics — dominate
        in count on catalog pages but aren't product photos)

    Returns the same shape as `calibrate_per_vendor`.
    """
    vendor_by_page = {r["page_no"]: (r.get("vendor") or "__unknown__")
                      for r in page_records}
    by_vendor: dict[str, dict] = {}
    all_widths: list[float] = []
    all_heights: list[float] = []
    doc_pages = 0
    for entry in pymupdf_pages:
        page_no = entry.get("page_no")
        vendor = vendor_by_page.get(page_no, "__unknown__")
        rects = entry.get("image_rects") or []
        widths: list[float] = []
        heights: list[float] = []
        for ir in rects:
            xref = int(ir.get("xref", -1))
            if (xref_pages is not None and recurring_threshold is not None
                    and xref_pages.get(xref, 0) >= recurring_threshold):
                continue  # skip decorative xrefs from the calibration sample
            bb = ir.get("bbox_px") or []
            if len(bb) != 4:
                continue
            w, h = _bbox_wh(bb)
            if w <= 0 or h <= 0:
                continue
            if w < min_size_px or h < min_size_px:
                continue  # tiny icon / swatch — not a product photo
            widths.append(w)
            heights.append(h)
        if not widths:
            continue
        slot = by_vendor.setdefault(vendor, {"widths": [], "heights": [],
                                             "pages": set()})
        slot["widths"].extend(widths)
        slot["heights"].extend(heights)
        slot["pages"].add(page_no)
        all_widths.extend(widths)
        all_heights.extend(heights)
        doc_pages += 1

    out: dict[str, dict] = {}
    for vendor, slot in by_vendor.items():
        s = _stats(slot["widths"], slot["heights"])
        s["n_pages"] = len(slot["pages"])
        out[vendor] = s
    doc = _stats(all_widths, all_heights)
    doc["n_pages"] = doc_pages
    out["__doc_wide__"] = doc
    return out


def calibrate_per_vendor(yolo_pages: list[dict],
                         page_records: list[dict]) -> dict:
    """Compute size/aspect priors for YOLO figure boxes grouped by vendor.

    Args:
      yolo_pages   : list of {page_no, boxes: [Box, ...]}
      page_records : list of {page_no, vendor, label} — vendor is the
                     `prev_context.vendor` (or `context_after.vendor`)
                     attached to each product page.

    Returns:
      {
        "<vendor>": {median_w, median_h, median_aspect, iqr_w, iqr_h,
                     iqr_aspect, n_pages, n_boxes},
        ...,
        "__doc_wide__": { ...same shape... }
      }

    Vendors with < 1 product page get no entry — callers fall back to
    `__doc_wide__`. The `n_pages` field lets callers decide between
    per-vendor and doc-wide tiers (planned threshold: >= 5).
    """
    vendor_by_page = {r["page_no"]: (r.get("vendor") or "__unknown__")
                      for r in page_records}
    by_vendor: dict[str, dict] = {}
    all_widths: list[float] = []
    all_heights: list[float] = []
    doc_pages = 0
    for entry in yolo_pages:
        page_no = entry.get("page_no")
        vendor = vendor_by_page.get(page_no, "__unknown__")
        figs = figure_boxes(entry.get("boxes") or [])
        if not figs:
            continue
        widths = []
        heights = []
        for b in figs:
            w, h = _bbox_wh(b["bbox"])
            if w <= 0 or h <= 0:
                continue
            widths.append(w)
            heights.append(h)
        if not widths:
            continue
        slot = by_vendor.setdefault(vendor, {"widths": [], "heights": [],
                                             "pages": set()})
        slot["widths"].extend(widths)
        slot["heights"].extend(heights)
        slot["pages"].add(page_no)
        all_widths.extend(widths)
        all_heights.extend(heights)
        doc_pages += 1

    out: dict[str, dict] = {}
    for vendor, slot in by_vendor.items():
        s = _stats(slot["widths"], slot["heights"])
        s["n_pages"] = len(slot["pages"])
        out[vendor] = s
    doc = _stats(all_widths, all_heights)
    doc["n_pages"] = doc_pages
    out["__doc_wide__"] = doc
    return out


def filter_candidates_by_size(figs: list[dict], stats: dict,
                              size_tol: float = 0.5,
                              aspect_tol: float = 0.5
                              ) -> list[dict]:
    """Filter figure boxes by w/h/aspect against the calibration stats.

    Soft thresholds:
      keep if  abs(log2(box_w / median_w)) <= size_tol  AND same for h
              AND abs(log2(box_aspect / median_aspect)) <= aspect_tol
      where size_tol=0.5 ~= a factor-of-1.4 band on each axis.

    Boxes with zero area or undefined stats fall through to "keep all"
    so we don't lose pages with very few figures.
    """
    mw, mh, ma = stats.get("median_w"), stats.get("median_h"), stats.get("median_aspect")
    if not mw or not mh or not ma:
        return [{"box": b, "kept": True, "reason": None} for b in figs]
    out = []
    for b in figs:
        w, h = _bbox_wh(b["bbox"])
        if w <= 0 or h <= 0:
            out.append({"box": b, "kept": False, "reason": "zero-area"})
            continue
        asp = w / h
        dw = abs(math.log2(w / mw)) if mw > 0 else 0
        dh = abs(math.log2(h / mh)) if mh > 0 else 0
        da = abs(math.log2(asp / ma)) if ma > 0 else 0
        if dw > size_tol or dh > size_tol:
            out.append({"box": b, "kept": False,
                        "reason": f"size {w:.0f}x{h:.0f} vs median {mw:.0f}x{mh:.0f}"})
        elif da > aspect_tol:
            out.append({"box": b, "kept": False,
                        "reason": f"aspect {asp:.2f} vs median {ma:.2f}"})
        else:
            out.append({"box": b, "kept": True, "reason": None})
    return out
