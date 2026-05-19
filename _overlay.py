"""Audit visualization for the YOLO + PyMuPDF + matching pipeline.

Two outputs:

  render_audit(fullres_png, yolo, pymupdf, match_log, out_png)
    Pillow ImageDraw overlay on top of the full-res page render. Every
    detection, filter decision, and SKU<->photo connection is drawn.

  render_index(pages_dir, out_html)
    Doc-wide static HTML index linking to every per-page audit PNG with
    a stats row. No JS framework — pure relative-link HTML.

The visual style spec is locked in CLAUDE.md / the change plan; tweak in
one place if it needs adjusting (see _STYLE below).
"""

from __future__ import annotations

import html
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# RGBA colors so we can overdraw without obliterating the page.
_STYLE = {
    "yolo_figure_kept":    (40, 90, 240, 220),   # blue
    "yolo_figure_dropped": (130, 130, 130, 180), # gray
    "yolo_text":           (40, 180, 90, 200),   # green
    "pymupdf_image":       (0, 200, 220, 220),   # cyan (dashed)
    "pymupdf_sku_word":    (220, 30, 200, 220),  # magenta
    "link_yolo":           (40, 90, 240, 200),   # thin blue
    "link_pymupdf":        (0, 200, 220, 200),   # thin cyan dashed
    "link_agree":          (40, 200, 90, 240),   # thick green
    "link_disagree":       (240, 200, 30, 240),  # yellow
    "link_low":            (240, 40, 40, 240),   # red
    "legend_bg":           (255, 255, 255, 235),
    "label_bg":            (255, 255, 255, 210),
    "label_fg":            (20, 20, 20, 255),
}


def _font(size: int = 14):
    # Built-in default font is portable across environments; no Windows
    # font-path bookkeeping needed.
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:  # noqa: BLE001
        return ImageFont.load_default()


def _draw_dashed_rect(draw: ImageDraw.ImageDraw, bbox, color, width=2,
                      dash=8, gap=6):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    # top + bottom
    for (sy, ey) in [(y1, y1), (y2, y2)]:
        x = x1
        on = True
        while x < x2:
            seg_end = min(x2, x + (dash if on else gap))
            if on:
                draw.line([(x, sy), (seg_end, ey)], fill=color, width=width)
            x = seg_end
            on = not on
    # left + right
    for (sx, ex) in [(x1, x1), (x2, x2)]:
        y = y1
        on = True
        while y < y2:
            seg_end = min(y2, y + (dash if on else gap))
            if on:
                draw.line([(sx, y), (ex, seg_end)], fill=color, width=width)
            y = seg_end
            on = not on


def _draw_label(draw: ImageDraw.ImageDraw, xy, text, font):
    if not text:
        return
    x, y = xy
    pad = 3
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
    except Exception:  # noqa: BLE001
        w, h = font.getbbox(text)[2:] if hasattr(font, "getbbox") else (8 * len(text), 14)
    box = [x, y, x + w + 2 * pad, y + h + 2 * pad]
    draw.rectangle(box, fill=_STYLE["label_bg"])
    draw.text((x + pad, y + pad), text, font=font, fill=_STYLE["label_fg"])


def _center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def render_audit(fullres_png: Path, yolo: dict | None,
                 pymupdf: dict | None, match_log: dict | None,
                 out_png: Path) -> Path:
    """Draw the audit overlay. All three input dicts are optional —
    callers passing partial state (e.g. probe Mode 1: yolo only) get
    a sensible subset of the visualization.

    yolo shape       : {"boxes": [{"cls", "bbox", "conf"}, ...]}
    pymupdf shape    : {"words": [...], "image_rects": [...]}
                       (the full output of _pymupdf_obs.observations)
    match_log shape  : {"filtered_photo_candidates": [...],
                        "skus": [{"sku", "localization", "match_combined", ...}],
                        "expected_pairing": {...}}
    """
    base = Image.open(fullres_png).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _font(13)
    small = _font(11)

    # ---- YOLO boxes -----------------------------------------------------
    if yolo:
        # Build a (id(box) -> filter result) lookup if match_log provided it.
        yolo_filter: dict[tuple, dict] = {}
        if match_log:
            for cand in match_log.get("filtered_photo_candidates", []) or []:
                if cand.get("source") != "yolo":
                    continue
                bb = tuple(cand.get("bbox_px") or [])
                yolo_filter[bb] = cand
        for b in yolo.get("boxes") or []:
            cls = (b.get("cls") or "").lower()
            bbox = b["bbox"]
            is_figure = ("figure" in cls) or cls == "image"
            is_text = (not is_figure) and any(t in cls for t in
                ("text", "title", "caption", "list", "header", "footer",
                 "abandon", "footnote"))
            tup = tuple(bbox)
            filt = yolo_filter.get(tup)
            if is_figure:
                if filt is not None and filt.get("kept") is False:
                    draw.rectangle(bbox, outline=_STYLE["yolo_figure_dropped"],
                                   width=2)
                    if filt.get("reason"):
                        _draw_label(draw, (bbox[0], bbox[3] + 2),
                                    f"drop: {filt['reason']}", small)
                else:
                    draw.rectangle(bbox, outline=_STYLE["yolo_figure_kept"],
                                   width=3)
                    _draw_label(draw, (bbox[0], max(0, bbox[1] - 16)),
                                f"figure {b.get('conf', 0):.2f}", small)
            elif is_text:
                draw.rectangle(bbox, outline=_STYLE["yolo_text"], width=2)

    # ---- PyMuPDF image rects --------------------------------------------
    if pymupdf:
        pymupdf_filter: dict[tuple, dict] = {}
        if match_log:
            for cand in match_log.get("filtered_photo_candidates", []) or []:
                if cand.get("source") != "pymupdf":
                    continue
                bb = tuple(cand.get("bbox_px") or [])
                pymupdf_filter[bb] = cand
        for ir in pymupdf.get("image_rects") or []:
            bbox = ir["bbox_px"]
            tup = tuple(bbox)
            filt = pymupdf_filter.get(tup)
            color = _STYLE["pymupdf_image"]
            _draw_dashed_rect(draw, bbox, color, width=2)
            label = f"xref:{ir['xref']}"
            if filt is not None and filt.get("kept") is False:
                label = f"{label} (drop: {filt.get('reason','')})"
            _draw_label(draw, (bbox[2] + 2, bbox[1]), label, small)

    # ---- SKU localization bboxes ----------------------------------------
    skus = (match_log or {}).get("skus", []) or []
    for s in skus:
        loc = s.get("localization") or {}
        if not loc:
            continue
        bbox = loc.get("bbox_px")
        if not bbox:
            continue
        draw.rectangle(bbox, outline=_STYLE["pymupdf_sku_word"], width=2)

    # ---- SKU<->photo links ---------------------------------------------
    for s in skus:
        loc = s.get("localization") or {}
        loc_bbox = loc.get("bbox_px")
        if not loc_bbox:
            continue
        sku_center = _center(loc_bbox)

        my = s.get("match_yolo_only") or {}
        mp = s.get("match_pymupdf_only") or {}
        mc = s.get("match_combined") or {}

        # Background per-detector lines (thin / dashed)
        if my.get("photo_bbox_px"):
            pc = _center(my["photo_bbox_px"])
            draw.line([sku_center, pc], fill=_STYLE["link_yolo"], width=1)
        if mp.get("photo_bbox_px"):
            pc = _center(mp["photo_bbox_px"])
            _draw_dashed_line(draw, sku_center, pc,
                              color=_STYLE["link_pymupdf"], width=1)

        # Foreground combined / final line
        if mc.get("photo_bbox_px"):
            pc = _center(mc["photo_bbox_px"])
            conf = (mc.get("confidence") or "").lower()
            agree = (mc.get("agreement") or "").lower()
            if conf == "low":
                col = _STYLE["link_low"]
            elif agree == "agree":
                col = _STYLE["link_agree"]
            else:
                col = _STYLE["link_disagree"]
            draw.line([sku_center, pc], fill=col, width=4)
            label = f"{s.get('sku','?')} [{conf.upper() or '?'}]"
            _draw_label(draw, (sku_center[0] + 6, sku_center[1] - 18),
                        label, font)
        else:
            label = f"{s.get('sku','?')} [LOC ONLY]"
            _draw_label(draw, (sku_center[0] + 6, sku_center[1] - 18),
                        label, font)

    # ---- Legend strip ---------------------------------------------------
    _draw_legend(draw, base.size, font)

    out = Image.alpha_composite(base, overlay).convert("RGB")
    out.save(out_png, format="PNG", optimize=True)
    return out_png


def _draw_dashed_line(draw, p1, p2, color, width=1, dash=10, gap=6):
    import math
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    dist = math.hypot(dx, dy)
    if dist == 0:
        return
    ux, uy = dx / dist, dy / dist
    pos = 0.0
    on = True
    while pos < dist:
        seg = min(dist - pos, dash if on else gap)
        if on:
            sx, sy = x1 + ux * pos, y1 + uy * pos
            ex, ey = x1 + ux * (pos + seg), y1 + uy * (pos + seg)
            draw.line([(sx, sy), (ex, ey)], fill=color, width=width)
        pos += seg
        on = not on


def _draw_legend(draw, size, font):
    w, _ = size
    pad = 8
    items = [
        ("YOLO figure (kept)",     _STYLE["yolo_figure_kept"], "rect"),
        ("YOLO figure (dropped)",  _STYLE["yolo_figure_dropped"], "rect"),
        ("YOLO text",              _STYLE["yolo_text"], "rect"),
        ("PyMuPDF image xref",     _STYLE["pymupdf_image"], "dash-rect"),
        ("SKU word bbox",          _STYLE["pymupdf_sku_word"], "rect"),
        ("Match: agree",           _STYLE["link_agree"], "line"),
        ("Match: disagree",        _STYLE["link_disagree"], "line"),
        ("Match: low confidence",  _STYLE["link_low"], "line"),
    ]
    x = pad
    y = pad
    h = 22
    box_w = 24
    draw.rectangle([0, 0, w, h + 2 * pad], fill=_STYLE["legend_bg"])
    for label, color, kind in items:
        if kind == "rect":
            draw.rectangle([x, y, x + box_w, y + h - 6], outline=color, width=2)
        elif kind == "dash-rect":
            _draw_dashed_rect(draw, [x, y, x + box_w, y + h - 6], color, width=2)
        else:
            draw.line([(x, y + (h - 6) // 2), (x + box_w, y + (h - 6) // 2)],
                      fill=color, width=4)
        try:
            tw = draw.textbbox((0, 0), label, font=font)[2]
        except Exception:  # noqa: BLE001
            tw = 9 * len(label)
        draw.text((x + box_w + 4, y), label, font=font,
                  fill=_STYLE["label_fg"])
        x += box_w + 8 + tw + 14


# ---- Doc-wide HTML index -------------------------------------------------


def _load_match(pages_dir: Path, page_no: int) -> dict | None:
    p = pages_dir / f"{page_no:02d}.match.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None


def _load_page(pages_dir: Path, page_no: int) -> dict | None:
    p = pages_dir / f"{page_no:02d}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None


def _page_stats(match: dict | None) -> dict:
    if not match:
        return {"n_skus": 0, "n_matched": 0, "high": 0, "medium": 0,
                "low": 0, "loc_failed": 0}
    skus = match.get("skus") or []
    high = sum(1 for s in skus if ((s.get("match_combined") or {}).get("confidence") == "high"))
    med = sum(1 for s in skus if ((s.get("match_combined") or {}).get("confidence") == "medium"))
    low = sum(1 for s in skus if ((s.get("match_combined") or {}).get("confidence") == "low"))
    matched = sum(1 for s in skus if (s.get("match_combined") or {}).get("photo_bbox_px"))
    loc_failed = sum(1 for s in skus if (s.get("localization") or {}).get("method") == "failed"
                     or not (s.get("localization") or {}).get("bbox_px"))
    return {"n_skus": len(skus), "n_matched": matched,
            "high": high, "medium": med, "low": low,
            "loc_failed": loc_failed}


def render_index(pages_dir: Path, out_html: Path,
                 calibration: dict | None = None) -> Path:
    """Write a static HTML index linking every <NN>.audit.png alongside
    per-page match stats and (optionally) the doc-wide calibration."""
    page_files = sorted(pages_dir.glob("*.audit.png"))
    rows: list[str] = []
    totals = {"n_skus": 0, "n_matched": 0, "high": 0, "medium": 0,
              "low": 0, "loc_failed": 0, "n_pages": 0}
    for png in page_files:
        try:
            page_no = int(png.stem.split(".")[0])
        except ValueError:
            continue
        match = _load_match(pages_dir, page_no)
        page = _load_page(pages_dir, page_no)
        stats = _page_stats(match)
        for k, v in stats.items():
            totals[k] += v
        totals["n_pages"] += 1
        vendor = ((page or {}).get("context_after") or
                  (page or {}).get("prev_context") or {}).get("vendor") or "?"
        rows.append(
            f"<div class=row>"
            f"<a href=\"{html.escape(png.name)}\">"
            f"<img loading=lazy src=\"{html.escape(png.name)}\"/></a>"
            f"<div class=meta>"
            f"<div class=pageno>page {page_no:02d}</div>"
            f"<div class=vendor>vendor: {html.escape(str(vendor))}</div>"
            f"<div>{stats['n_skus']} SKUs, {stats['n_matched']} matched</div>"
            f"<div class=conf>"
            f"<span class=high>{stats['high']} high</span>&nbsp;"
            f"<span class=medium>{stats['medium']} med</span>&nbsp;"
            f"<span class=low>{stats['low']} low</span>"
            f"</div>"
            f"<div>loc-failed: {stats['loc_failed']}</div>"
            f"</div>"
            f"</div>"
        )

    calib_block = ""
    if calibration:
        items: list[str] = []
        for vendor, stats in calibration.items():
            if not isinstance(stats, dict):
                continue
            mw = stats.get("median_w")
            mh = stats.get("median_h")
            ma = stats.get("median_aspect")
            np_ = stats.get("n_pages")
            nb = stats.get("n_boxes")
            items.append(
                f"<tr><td>{html.escape(str(vendor))}</td>"
                f"<td>{np_}</td><td>{nb}</td>"
                f"<td>{('%.0f' % mw) if mw else '-'}</td>"
                f"<td>{('%.0f' % mh) if mh else '-'}</td>"
                f"<td>{('%.2f' % ma) if ma else '-'}</td>"
                f"</tr>"
            )
        calib_block = (
            "<h2>Per-vendor calibration</h2>"
            "<table><thead><tr>"
            "<th>vendor</th><th>pages</th><th>figs</th>"
            "<th>med w</th><th>med h</th><th>med aspect</th>"
            "</tr></thead><tbody>"
            + "".join(items)
            + "</tbody></table>"
        )

    agreement_block = (
        f"<div>Pages: {totals['n_pages']} | "
        f"SKUs: {totals['n_skus']} | matched: {totals['n_matched']} | "
        f"<span class=high>high: {totals['high']}</span> | "
        f"<span class=medium>medium: {totals['medium']}</span> | "
        f"<span class=low>low: {totals['low']}</span> | "
        f"loc-failed: {totals['loc_failed']}</div>"
    )

    css = """
      body { font-family: -apple-system, sans-serif; margin: 16px; }
      h1 { font-size: 18px; }
      .row { display: flex; gap: 12px; margin: 12px 0; padding: 8px;
             border: 1px solid #ddd; border-radius: 6px; }
      .row img { max-width: 480px; max-height: 360px; object-fit: contain; }
      .meta { font-size: 13px; line-height: 1.5; }
      .pageno { font-weight: 600; }
      .vendor { color: #555; }
      .high { color: #2e8b57; font-weight: 600; }
      .medium { color: #b8860b; }
      .low { color: #b22222; font-weight: 600; }
      table { border-collapse: collapse; margin: 8px 0; font-size: 13px; }
      th, td { border: 1px solid #ccc; padding: 4px 8px; text-align: left; }
      thead th { background: #f4f4f4; }
    """

    body = (
        f"<!doctype html><meta charset=utf-8><title>Audit — {pages_dir.name}</title>"
        f"<style>{css}</style>"
        f"<h1>Audit — {html.escape(pages_dir.name)}</h1>"
        f"{agreement_block}"
        f"{calib_block}"
        f"<h2>Pages</h2>"
        + "".join(rows)
    )
    out_html.write_text(body, encoding="utf-8")
    return out_html
