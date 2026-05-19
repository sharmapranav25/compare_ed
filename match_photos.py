"""Step 3 (new): match each extracted SKU to its product photo using
YOLO + PyMuPDF observations gathered during extract.

Inputs (already on disk):
  <doc>.pages/<NN>.json          — products[] from extract_products.py
  <doc>.pages/<NN>.yolo.json     — figure/text boxes
  <doc>.pages/<NN>.pymupdf.json  — words + image xref rects
  <doc>.pages/<NN>.fullres.png   — for the audit overlay

Outputs:
  <doc>.pages/<NN>.match.json    — full per-SKU decision log
  <doc>.pages/<NN>.audit.png     — overlay visualization
  <doc>.pages/<NN>_<sku>.png     — per-SKU photo crop (optional)
  <doc>.pages/_calibration.json  — doc-wide stats blob (per-vendor)
  <doc>.pages/_audit_index.html  — browseable index

Each product entry in <NN>.json is updated in-place with photo_box,
photo_box_source, match_confidence, photo_crop.

Algorithm in 8 lines:
  1. Aggregate yolo + pymupdf observations across all product pages.
  2. Compute doc-wide xref page-counts → flag recurring decorative xrefs.
  3. Compute per-vendor YOLO figure-box size/aspect priors.
  4. Pass 1 — per page, naive nearest-neighbor on size/aspect-filtered
              candidates → collect (SKU, photo) pairings where YOLO and
              PyMuPDF agreed.
  5. Calibrate per-vendor distance + direction from pass-1 pairings.
  6. Pass 2 — re-match every SKU with soft scoring (distance + direction)
              over filtered candidates. Run three streams: yolo_only,
              pymupdf_only, combined.
  7. Persist <NN>.match.json + update <NN>.json product fields.
  8. Render audit PNGs + HTML index.

Usage:
  python match_photos.py files/<vendor>/<doc>.pdf
  python match_photos.py files/<vendor>/<doc>.pdf --force
  python match_photos.py files/<vendor>/<doc>.pdf --page 7
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _layout import (  # noqa: E402
    calibrate_per_vendor,
    calibrate_pymupdf_per_vendor,
    figure_boxes,
    filter_candidates_by_size,
)
from _overlay import render_audit, render_index  # noqa: E402
from _pymupdf_obs import find_sku_word_bbox  # noqa: E402

# --- Constants -------------------------------------------------------------

RECURRING_XREF_FRACTION = 0.6        # xref on > 60% of pages = decorative
SIZE_TOL = 0.6                        # log2 — ~factor-of-1.5 size band (YOLO only)
ASPECT_TOL = 0.6
PYMUPDF_MIN_DIM_PX = 40               # rects below this on either dim = swatch/icon
PYMUPDF_MAX_ASPECT = 5.0              # anything wider than 5:1 = banner, not a shoe
PER_VENDOR_MIN_PAGES = 5
BOOTSTRAP_MIN_PAIRS = 5               # below this → no calibration, NN only
AGREE_IOU = 0.5
DIRECTION_WEIGHT = 0.5
HIGH_SCORE = 2.0                      # total_score ≤ HIGH_SCORE → high
MEDIUM_SCORE = 8.0                    # total_score ≤ MEDIUM_SCORE → medium

# --- Geometry helpers ------------------------------------------------------


def _center(bb: list[float]) -> tuple[float, float]:
    return ((bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0)


def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a_area = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    b_area = max(0.0, (bx2 - bx1) * (by2 - by1))
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


def _distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def _angle(src: tuple[float, float], dst: tuple[float, float]) -> float:
    """Angle of (dst - src) vector in radians, in [-pi, pi]."""
    return math.atan2(dst[1] - src[1], dst[0] - src[0])


def _circmean(angles: list[float]) -> float:
    if not angles:
        return 0.0
    s = sum(math.sin(a) for a in angles)
    c = sum(math.cos(a) for a in angles)
    return math.atan2(s, c)


def _circconcentration(angles: list[float]) -> float:
    """Mean resultant length ∈ [0, 1]. 1 = perfectly concentrated."""
    if not angles:
        return 0.0
    s = sum(math.sin(a) for a in angles) / len(angles)
    c = sum(math.cos(a) for a in angles) / len(angles)
    return math.hypot(s, c)


def _iqr(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    vs = sorted(values)
    n = len(vs)
    q1 = vs[max(0, n // 4)]
    q3 = vs[min(n - 1, (3 * n) // 4)]
    return max(0.0, q3 - q1)


# --- Page-data loading -----------------------------------------------------


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return None


def _vendor_for_page(page: dict) -> str:
    ctx = page.get("prev_context") or page.get("context_after") or {}
    return (ctx.get("vendor") or "__unknown__")


def collect_pages(pages_dir: Path) -> list[dict]:
    """Returns a list of {page_no, page, yolo, pymupdf, fullres} for every
    page with label='product' (and products in its JSON). Sorted by page_no."""
    out: list[dict] = []
    for json_path in sorted(pages_dir.glob("*.json")):
        # Skip our own sibling jsons (yolo/pymupdf/match) by basename pattern
        stem = json_path.stem
        if any(stem.endswith(s) for s in (".yolo", ".pymupdf", ".match")):
            continue
        page = _load_json(json_path)
        if not page:
            continue
        if page.get("label") != "product":
            continue
        page_no = page.get("page_no")
        yolo = _load_json(pages_dir / f"{page_no:02d}.yolo.json")
        pymupdf = _load_json(pages_dir / f"{page_no:02d}.pymupdf.json")
        fullres = pages_dir / f"{page_no:02d}.fullres.png"
        out.append({"page_no": page_no, "page": page, "page_path": json_path,
                    "yolo": yolo, "pymupdf": pymupdf,
                    "fullres": fullres if fullres.exists() else None})
    return out


# --- Doc-wide xref decorative filter ---------------------------------------


def compute_recurring_xrefs(records: list[dict]) -> dict[int, int]:
    """xref -> number of pages it appears on. Caller decides threshold."""
    count: dict[int, int] = {}
    for rec in records:
        seen = set()
        for ir in ((rec["pymupdf"] or {}).get("image_rects") or []):
            xref = ir.get("xref")
            if xref is None:
                continue
            seen.add(int(xref))
        for x in seen:
            count[x] = count.get(x, 0) + 1
    return count


# --- Per-page candidate filtering ------------------------------------------


def _filter_yolo(boxes: list[dict], stats: dict) -> list[dict]:
    """Return [{box, kept, reason}, ...] for figure boxes only."""
    figs = figure_boxes(boxes)
    return filter_candidates_by_size(figs, stats,
                                     size_tol=SIZE_TOL,
                                     aspect_tol=ASPECT_TOL)


def _filter_pymupdf(image_rects: list[dict],
                    xref_pages: dict[int, int],
                    n_pages: int,
                    stats: dict) -> list[dict]:
    """Returns [{rect, kept, reason}, ...].

    PyMuPDF rects are exact PDF image objects, so we trust them. Two
    cheap sanity checks only:
      - both dims >= PYMUPDF_MIN_DIM_PX (kills swatches / color dots)
      - aspect ratio <= PYMUPDF_MAX_ASPECT (kills header banners)
    Plus the recurring-xref decorative filter.

    `stats` is accepted but unused — kept in the signature for caller
    compatibility while we A/B the median-band approach.
    """
    del stats
    out = []
    thresh = max(2, math.ceil(n_pages * RECURRING_XREF_FRACTION))
    for ir in image_rects:
        xref = int(ir.get("xref", -1))
        bb = ir.get("bbox_px") or []
        if len(bb) != 4:
            out.append({"rect": ir, "kept": False, "reason": "no bbox"})
            continue
        w = bb[2] - bb[0]
        h = bb[3] - bb[1]
        if w <= 0 or h <= 0:
            out.append({"rect": ir, "kept": False, "reason": "zero area"})
            continue
        page_count = xref_pages.get(xref, 0)
        if xref >= 0 and page_count >= thresh and n_pages >= 3:
            out.append({"rect": ir, "kept": False,
                        "reason": f"xref on {page_count}/{n_pages} pages — recurring decorative"})
            continue
        if w < PYMUPDF_MIN_DIM_PX or h < PYMUPDF_MIN_DIM_PX:
            out.append({"rect": ir, "kept": False,
                        "reason": f"size {w:.0f}x{h:.0f} below min {PYMUPDF_MIN_DIM_PX}px"})
            continue
        aspect = max(w / h, h / w)
        if aspect > PYMUPDF_MAX_ASPECT:
            out.append({"rect": ir, "kept": False,
                        "reason": f"aspect {aspect:.1f} > {PYMUPDF_MAX_ASPECT}:1 (banner)"})
            continue
        out.append({"rect": ir, "kept": True, "reason": None})
    return out


# --- SKU localization ------------------------------------------------------


def localize_skus(products: list[dict], pymupdf: dict | None
                  ) -> list[tuple[dict, dict]]:
    """Returns [(product, localization_dict), ...] aligned with `products`.

    localization_dict is one of:
      {"method": "pymupdf_text", "bbox_px": [...]}   — found in word list
      {"method": "failed"}                            — not findable
    """
    words = (pymupdf or {}).get("words") or []
    out: list[tuple[dict, dict]] = []
    for p in products:
        sku = (p.get("sku") or "").strip()
        bbox = find_sku_word_bbox(words, sku) if (sku and words) else None
        if bbox:
            out.append((p, {"method": "pymupdf_text", "bbox_px": bbox}))
        else:
            out.append((p, {"method": "failed"}))
    return out


# --- Matching --------------------------------------------------------------


def _nearest(candidates: list[list[float]], sku_xy: tuple[float, float]
             ) -> tuple[int, float] | None:
    """Returns (index, distance) of the candidate whose center is nearest
    to sku_xy. None if no candidates."""
    best = None
    for i, bb in enumerate(candidates):
        d = _distance(sku_xy, _center(bb))
        if best is None or d < best[1]:
            best = (i, d)
    return best


def _score(distance: float, angle: float, calib: dict | None) -> float:
    """Soft total score combining distance and direction against a
    per-vendor (or doc-wide) calibration. Lower = better fit."""
    if not calib or calib.get("expected_distance") is None:
        return distance  # raw distance fallback
    exp_d = calib["expected_distance"]
    sigma = calib.get("sigma_distance") or max(1.0, exp_d * 0.25)
    exp_a = calib.get("expected_angle", 0.0)
    dist_z = ((distance - exp_d) / max(1.0, sigma)) ** 2
    if calib.get("angle_concentration", 0.0) >= 0.5:
        dir_pen = 1.0 - math.cos(angle - exp_a)
    else:
        dir_pen = 0.0  # vendor has no clear directional preference
    return dist_z + DIRECTION_WEIGHT * dir_pen


def _confidence_label(score: float) -> str:
    if score <= HIGH_SCORE:
        return "high"
    if score <= MEDIUM_SCORE:
        return "medium"
    return "low"


def _agreement(yolo_bb: list[float] | None,
               pymupdf_bb: list[float] | None) -> str:
    if yolo_bb is None and pymupdf_bb is None:
        return "neither"
    if yolo_bb is None:
        return "pymupdf_only"
    if pymupdf_bb is None:
        return "yolo_only"
    return "agree" if _iou(yolo_bb, pymupdf_bb) >= AGREE_IOU else "disagree"


def _combined_pick(yolo_bb: list[float] | None,
                   yolo_score: float | None,
                   pymupdf_bb: list[float] | None,
                   pymupdf_score: float | None,
                   ) -> tuple[list[float] | None, str, str, float | None]:
    """Choose the combined match given each stream's pick.

    Returns (chosen_bbox, source, agreement, score)."""
    agreement = _agreement(yolo_bb, pymupdf_bb)
    if agreement == "agree":
        # Prefer pymupdf (exact PDF object) on agreement
        return pymupdf_bb, "both", "agree", min(
            s for s in (yolo_score, pymupdf_score) if s is not None
        )
    if agreement == "yolo_only":
        return yolo_bb, "yolo_only", "yolo_only", yolo_score
    if agreement == "pymupdf_only":
        return pymupdf_bb, "pymupdf_only", "pymupdf_only", pymupdf_score
    if agreement == "disagree":
        # Pick the lower-scoring one but mark disagreement
        if yolo_score is None:
            return pymupdf_bb, "pymupdf_only", "disagree", pymupdf_score
        if pymupdf_score is None:
            return yolo_bb, "yolo_only", "disagree", yolo_score
        if yolo_score <= pymupdf_score:
            return yolo_bb, "yolo_only", "disagree", yolo_score
        return pymupdf_bb, "pymupdf_only", "disagree", pymupdf_score
    return None, "none", "neither", None


def _kept_yolo_bboxes(yolo_filter: list[dict]) -> list[list[float]]:
    return [f["box"]["bbox"] for f in yolo_filter if f["kept"]]


def _kept_pymupdf_bboxes(pymupdf_filter: list[dict]) -> list[list[float]]:
    return [f["rect"]["bbox_px"] for f in pymupdf_filter if f["kept"]]


def match_pass1(records: list[dict],
                size_stats_per_vendor: dict,
                pymupdf_size_stats: dict,
                xref_pages: dict[int, int],
                n_total_pages: int) -> dict[str, list[dict]]:
    """Pass 1 — naive nearest-neighbor; return SKU<->photo pairings that
    both streams agreed on, grouped by vendor.

    Returns: {vendor: [{distance_px, angle_rad, page_no}, ...]}
    """
    pairings: dict[str, list[dict]] = {}
    for rec in records:
        page = rec["page"]
        vendor = _vendor_for_page(page)
        yolo_stats = size_stats_per_vendor.get(vendor) or size_stats_per_vendor.get(
            "__doc_wide__") or {}
        pdf_stats = pymupdf_size_stats.get(vendor) or pymupdf_size_stats.get(
            "__doc_wide__") or {}
        yolo_boxes = (rec["yolo"] or {}).get("boxes") or []
        pymupdf_rects = (rec["pymupdf"] or {}).get("image_rects") or []

        y_filter = _filter_yolo(yolo_boxes, yolo_stats)
        p_filter = _filter_pymupdf(pymupdf_rects, xref_pages,
                                   n_total_pages, pdf_stats)
        y_keep = _kept_yolo_bboxes(y_filter)
        p_keep = _kept_pymupdf_bboxes(p_filter)

        loc = localize_skus(page.get("products") or [], rec["pymupdf"])
        for _, loc_info in loc:
            if loc_info.get("method") == "failed":
                continue
            sku_xy = _center(loc_info["bbox_px"])
            y_nearest = _nearest(y_keep, sku_xy) if y_keep else None
            p_nearest = _nearest(p_keep, sku_xy) if p_keep else None
            if y_nearest is None or p_nearest is None:
                continue
            y_bb = y_keep[y_nearest[0]]
            p_bb = p_keep[p_nearest[0]]
            if _iou(y_bb, p_bb) < AGREE_IOU:
                continue
            chosen_bb = p_bb  # they agree — keep the exact one
            d = _distance(sku_xy, _center(chosen_bb))
            a = _angle(sku_xy, _center(chosen_bb))
            pairings.setdefault(vendor, []).append({
                "distance_px": d, "angle_rad": a, "page_no": rec["page_no"]
            })
    return pairings


def calibrate_distance_direction(pairings: dict[str, list[dict]]) -> dict:
    """Convert per-vendor pass-1 pairings into per-vendor distance +
    direction calibration. Vendors with too few pairs get None entries
    (caller falls back)."""
    out: dict[str, dict] = {}
    all_d: list[float] = []
    all_a: list[float] = []
    for vendor, ps in pairings.items():
        ds = [p["distance_px"] for p in ps]
        as_ = [p["angle_rad"] for p in ps]
        all_d.extend(ds)
        all_a.extend(as_)
        if len(ps) < BOOTSTRAP_MIN_PAIRS:
            out[vendor] = {"expected_distance": None,
                           "sigma_distance": None,
                           "expected_angle": None,
                           "angle_concentration": None,
                           "n_pairs": len(ps)}
            continue
        m = median(ds)
        sigma = _iqr(ds) / 1.349 if len(ds) > 1 else max(1.0, m * 0.25)
        out[vendor] = {
            "expected_distance": float(m),
            "sigma_distance": float(max(1.0, sigma)),
            "expected_angle": float(_circmean(as_)),
            "angle_concentration": float(_circconcentration(as_)),
            "n_pairs": len(ps),
        }
    if all_d:
        m = median(all_d)
        sigma = _iqr(all_d) / 1.349 if len(all_d) > 1 else max(1.0, m * 0.25)
        out["__doc_wide__"] = {
            "expected_distance": float(m),
            "sigma_distance": float(max(1.0, sigma)),
            "expected_angle": float(_circmean(all_a)),
            "angle_concentration": float(_circconcentration(all_a)),
            "n_pairs": len(all_d),
        }
    else:
        out["__doc_wide__"] = {"expected_distance": None,
                               "sigma_distance": None,
                               "expected_angle": None,
                               "angle_concentration": None,
                               "n_pairs": 0}
    return out


def _calib_tier(vendor: str,
                size_stats: dict,
                pymupdf_size_stats: dict,
                dist_stats: dict) -> tuple[str, dict, dict, dict]:
    """Pick which calibration tier to use for this vendor.
    Returns (tier_name, yolo_size_used, pymupdf_size_used, dist_used)."""
    s = size_stats.get(vendor)
    p = pymupdf_size_stats.get(vendor)
    d = dist_stats.get(vendor)
    if s and s.get("n_pages", 0) >= PER_VENDOR_MIN_PAGES:
        return ("per_vendor", s,
                p or pymupdf_size_stats.get("__doc_wide__", {}) or {},
                d or {})
    return ("doc_wide",
            size_stats.get("__doc_wide__", {}) or {},
            pymupdf_size_stats.get("__doc_wide__", {}) or {},
            dist_stats.get("__doc_wide__", {}) or {})


def _filter_record_for_log(y_filter, p_filter) -> list[dict]:
    log: list[dict] = []
    for f in y_filter:
        log.append({"source": "yolo", "bbox_px": f["box"]["bbox"],
                    "kept": f["kept"], "reason": f.get("reason"),
                    "cls": f["box"].get("cls"),
                    "conf": f["box"].get("conf")})
    for f in p_filter:
        log.append({"source": "pymupdf", "bbox_px": f["rect"]["bbox_px"],
                    "kept": f["kept"], "reason": f.get("reason"),
                    "xref": f["rect"].get("xref")})
    return log


def match_pass2(rec: dict, size_stats_per_vendor: dict,
                pymupdf_size_stats: dict,
                dist_stats_per_vendor: dict, xref_pages: dict[int, int],
                n_total_pages: int) -> dict:
    """Pass 2 — full matching for ONE page. Returns the <NN>.match.json
    payload."""
    page = rec["page"]
    vendor = _vendor_for_page(page)
    tier, yolo_size_used, pdf_size_used, dist_used = _calib_tier(
        vendor, size_stats_per_vendor, pymupdf_size_stats,
        dist_stats_per_vendor,
    )

    yolo_boxes = (rec["yolo"] or {}).get("boxes") or []
    pymupdf_rects = (rec["pymupdf"] or {}).get("image_rects") or []
    y_filter = _filter_yolo(yolo_boxes, yolo_size_used)
    p_filter = _filter_pymupdf(pymupdf_rects, xref_pages,
                               n_total_pages, pdf_size_used)
    y_keep = _kept_yolo_bboxes(y_filter)
    p_keep = _kept_pymupdf_bboxes(p_filter)

    loc_pairs = localize_skus(page.get("products") or [], rec["pymupdf"])

    skus_out: list[dict] = []
    for product, loc in loc_pairs:
        sku = product.get("sku") or ""
        if loc.get("method") == "failed":
            skus_out.append({"sku": sku, "localization": {"method": "failed"},
                             "match_yolo_only": None,
                             "match_pymupdf_only": None,
                             "match_combined": None})
            continue
        sku_xy = _center(loc["bbox_px"])

        # Stream 1 — YOLO
        my = None
        if y_keep:
            best = None
            for bb in y_keep:
                d = _distance(sku_xy, _center(bb))
                a = _angle(sku_xy, _center(bb))
                sc = _score(d, a, dist_used)
                if best is None or sc < best[2]:
                    best = (bb, d, sc, a)
            if best is not None:
                bb, d, sc, a = best
                my = {"photo_bbox_px": bb, "distance_px": d,
                      "angle_deg": math.degrees(a), "score": sc}

        # Stream 2 — PyMuPDF
        mp = None
        if p_keep:
            best = None
            for bb in p_keep:
                d = _distance(sku_xy, _center(bb))
                a = _angle(sku_xy, _center(bb))
                sc = _score(d, a, dist_used)
                if best is None or sc < best[2]:
                    best = (bb, d, sc, a)
            if best is not None:
                bb, d, sc, a = best
                mp = {"photo_bbox_px": bb, "distance_px": d,
                      "angle_deg": math.degrees(a), "score": sc}

        # Combined
        y_bb = my["photo_bbox_px"] if my else None
        p_bb = mp["photo_bbox_px"] if mp else None
        chosen, source, agreement, combined_score = _combined_pick(
            y_bb, my["score"] if my else None,
            p_bb, mp["score"] if mp else None,
        )
        confidence = None
        if chosen is not None:
            confidence = _confidence_label(combined_score
                                           if combined_score is not None
                                           else float("inf"))
            if agreement == "disagree" and confidence == "high":
                confidence = "medium"  # downgrade on disagreement

        mc = None
        if chosen is not None:
            mc_d = _distance(sku_xy, _center(chosen))
            mc_a = _angle(sku_xy, _center(chosen))
            mc = {"photo_bbox_px": chosen, "source": source,
                  "confidence": confidence, "agreement": agreement,
                  "distance_px": mc_d,
                  "angle_deg": math.degrees(mc_a),
                  "score": combined_score}

        skus_out.append({"sku": sku, "localization": loc,
                         "match_yolo_only": my,
                         "match_pymupdf_only": mp,
                         "match_combined": mc})

    return {
        "page_no": rec["page_no"],
        "vendor_used": vendor,
        "calibration_tier": tier,
        "calibration_used": {
            "yolo_size": yolo_size_used,
            "pymupdf_size": pdf_size_used,
            "distance": dist_used,
        },
        "filtered_photo_candidates": _filter_record_for_log(y_filter, p_filter),
        "skus": skus_out,
    }


# --- Persistence -----------------------------------------------------------


_SKU_SLUG_RE = re.compile(r"[^A-Za-z0-9]+")


def _sku_slug(sku: str) -> str:
    return _SKU_SLUG_RE.sub("_", sku).strip("_") or "sku"


def _maybe_crop(fullres_path: Path | None, bbox: list[float],
                out_path: Path) -> Path | None:
    if fullres_path is None or not fullres_path.exists():
        return None
    try:
        from PIL import Image
        img = Image.open(fullres_path)
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img.size[0], x2)
        y2 = min(img.size[1], y2)
        if x2 <= x1 or y2 <= y1:
            return None
        img.crop((x1, y1, x2, y2)).save(out_path, format="PNG", optimize=True)
        return out_path
    except Exception:  # noqa: BLE001
        return None


def update_page_json(page_path: Path, page: dict, match: dict,
                     pages_dir: Path, fullres_path: Path | None) -> None:
    """Merge match results back into <NN>.json — adds per-product photo
    fields and writes a per-SKU crop next to the page when matched."""
    skus_by_sku = {s["sku"]: s for s in match["skus"]}
    new_products: list[dict] = []
    page_no = page.get("page_no")
    for p in page.get("products") or []:
        s = skus_by_sku.get(p.get("sku") or "")
        new = dict(p)
        if not s:
            new["photo_box"] = None
            new["photo_box_source"] = None
            new["match_confidence"] = None
            new["photo_crop"] = None
            new_products.append(new)
            continue
        mc = s.get("match_combined")
        if mc and mc.get("photo_bbox_px"):
            bbox = mc["photo_bbox_px"]
            new["photo_box"] = {
                "x": float(bbox[0]), "y": float(bbox[1]),
                "w": float(bbox[2] - bbox[0]),
                "h": float(bbox[3] - bbox[1]),
            }
            new["photo_box_source"] = mc.get("source")
            new["match_confidence"] = mc.get("confidence")
            crop_name = f"{page_no:02d}_{_sku_slug(p.get('sku') or '')}.png"
            crop = _maybe_crop(fullres_path, bbox, pages_dir / crop_name)
            new["photo_crop"] = crop.name if crop else None
        else:
            new["photo_box"] = None
            new["photo_box_source"] = None
            new["match_confidence"] = None
            new["photo_crop"] = None
        new_products.append(new)
    page["products"] = new_products
    page_path.write_text(json.dumps(page, indent=2))


# --- Driver ----------------------------------------------------------------


def match_pdf(pdf_path: Path, only_page: int | None, force: bool) -> None:
    pages_dir = pdf_path.with_name(pdf_path.stem + ".pages")
    if not pages_dir.exists():
        print(f"error: {pages_dir} not found — run classify + extract first",
              file=sys.stderr)
        sys.exit(1)
    records = collect_pages(pages_dir)
    if not records:
        print("no product pages with extractions found; nothing to match",
              file=sys.stderr)
        return
    if all(r["yolo"] is None and r["pymupdf"] is None for r in records):
        print("no yolo/pymupdf observations found — re-run extract with "
              "doclayout-yolo installed", file=sys.stderr)
        return

    # 1. Per-vendor size calibrations — YOLO and PyMuPDF computed
    #    independently. (YOLO sometimes collapses dense grids into one
    #    huge figure; using its median to size-filter PyMuPDF rects then
    #    rejects all the real photos.)
    yolo_pages = [{"page_no": r["page_no"],
                   "boxes": (r["yolo"] or {}).get("boxes") or []}
                  for r in records]
    pymupdf_pages = [{"page_no": r["page_no"],
                      "image_rects": (r["pymupdf"] or {}).get("image_rects") or []}
                     for r in records]
    page_records = [{"page_no": r["page_no"],
                     "vendor": _vendor_for_page(r["page"]),
                     "label": r["page"].get("label")} for r in records]
    size_stats = calibrate_per_vendor(yolo_pages, page_records)

    # 2. Doc-wide xref decorative counts
    xref_pages = compute_recurring_xrefs(records)
    n_total_pages = len(records)
    recurring_thresh = max(2, math.ceil(n_total_pages * RECURRING_XREF_FRACTION))
    pymupdf_size_stats = calibrate_pymupdf_per_vendor(
        pymupdf_pages, page_records,
        xref_pages=xref_pages,
        recurring_threshold=recurring_thresh if n_total_pages >= 3 else None,
    )

    # 3. Pass 1 — bootstrap pairings
    pairings = match_pass1(records, size_stats, pymupdf_size_stats,
                           xref_pages, n_total_pages)
    dist_stats = calibrate_distance_direction(pairings)

    calibration_blob = {
        "size_per_vendor": size_stats,
        "pymupdf_size_per_vendor": pymupdf_size_stats,
        "distance_per_vendor": dist_stats,
        "n_recurring_xrefs": sum(
            1 for c in xref_pages.values()
            if c >= max(2, math.ceil(n_total_pages * RECURRING_XREF_FRACTION))
        ),
        "n_pages": n_total_pages,
    }
    (pages_dir / "_calibration.json").write_text(
        json.dumps(calibration_blob, indent=2)
    )

    # 4. Pass 2 — full match per page
    target_pages = {only_page} if only_page else None
    n_done = 0
    n_skipped = 0
    for rec in records:
        if target_pages and rec["page_no"] not in target_pages:
            continue
        match_path = pages_dir / f"{rec['page_no']:02d}.match.json"
        if match_path.exists() and not force:
            n_skipped += 1
            continue
        match = match_pass2(rec, size_stats, pymupdf_size_stats,
                            dist_stats, xref_pages, n_total_pages)
        match_path.write_text(json.dumps(match, indent=2))
        update_page_json(rec["page_path"], rec["page"], match, pages_dir,
                         rec["fullres"])
        if rec["fullres"]:
            out_png = pages_dir / f"{rec['page_no']:02d}.audit.png"
            try:
                render_audit(rec["fullres"], rec["yolo"], rec["pymupdf"],
                             match, out_png)
            except Exception as exc:  # noqa: BLE001
                print(f"  page {rec['page_no']:02d}: overlay failed "
                      f"({type(exc).__name__}: {exc})", file=sys.stderr)
        n_done += 1
        print(f"  page {rec['page_no']:02d}: matched "
              f"({sum(1 for s in match['skus'] if (s['match_combined'] or {}).get('photo_bbox_px'))}/"
              f"{len(match['skus'])} skus) — tier={match['calibration_tier']}",
              file=sys.stderr)

    # 5. HTML index (always refresh — it's cheap)
    index_path = pages_dir / "_audit_index.html"
    try:
        render_index(pages_dir, index_path, calibration=size_stats)
    except Exception as exc:  # noqa: BLE001
        print(f"index render failed ({type(exc).__name__}: {exc})",
              file=sys.stderr)

    print(f"\nMATCH SUMMARY: {n_done} pages matched, {n_skipped} cached, "
          f"index -> {index_path}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--force", action="store_true",
                    help="Re-match pages even if <NN>.match.json exists")
    ap.add_argument("--page", type=int, default=None,
                    help="Only process this 1-indexed page")
    args = ap.parse_args()
    if not args.pdf.exists():
        ap.error(f"PDF not found: {args.pdf}")
    match_pdf(args.pdf, args.page, args.force)


if __name__ == "__main__":
    main()
