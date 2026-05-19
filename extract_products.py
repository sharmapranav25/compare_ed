"""Step 2 of agentic per-page parsing: extract products from `product` pages.

Reads <doc>.pages/<NN>.json from classify_pages.py. For each page with
label == "product", does a TWO-STAGE pass:

  Stage 1 — Opus 4.7 visual extract (recall):
    The VLM looks at the page + carried context (vendor, current_section)
    and returns every distinct product as a candidate row with all fields
    (sku, description, color, cost, retail, intro_date, gender_hint).
    This stage's job is to find everything — including odd / off-pattern
    SKUs and multi-brand catalogs where formats vary on the same page.

  Stage 2 — Sonnet 4.6 text-only SKU validator (precision):
    Given JUST the list of candidate SKU strings (no image), asks
    "for each, does this string look like a real product SKU, or is it
    a price / size / section header / false positive?" Returns yes/no
    per string. Candidates flagged "no" are dropped from the products
    list and moved to `rejected_candidates` for debugging.

Why this shape: the visual heavy-lifting stays where it belongs (the VLM
sees the page). The cheap text-only second pass exists to catch obvious
false positives in the SKU column — things like "MEN'S BOOTS" or "$120"
that occasionally get mis-extracted as a SKU. It cannot add SKUs the VLM
missed; it can only filter the candidate list.

Appends `products: [...]` and `rejected_candidates: [...]` to the same
<NN>.json. If any step errors, writes `error: "<reason>"` so
build_buysheet can flag the page. Skip pages where `products` already
exists unless --force.

Parallel: --workers N (default 5) fans out the per-page pair via
ThreadPoolExecutor. Each page is self-contained.

Usage:
  python agentic_parsing/extract_products.py files/<vendor>/<doc>.pdf
  python agentic_parsing/extract_products.py files/<vendor>/<doc>.pdf --workers 8
  python agentic_parsing/extract_products.py files/<vendor>/<doc>.pdf --model sonnet
  python agentic_parsing/extract_products.py files/<vendor>/<doc>.pdf --page 12 --force
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic
import fitz
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _render import (  # noqa: E402
    ImageTooLargeError, find_cached_render, media_type_for, render_page_safe,
)
from analysis.usage import usage_from_response  # noqa: E402
from detect import shoe as shoe_detect  # noqa: E402
from deterministic_check.verify import verify_against_text_layer  # noqa: E402

load_dotenv()

MODEL_OPUS = "claude-opus-4-7"
MODEL_SONNET = "claude-sonnet-4-6"

EXTRACT_SYSTEM = """You extract product rows from ONE page of a wholesale shoe buy-sheet PDF.

You will be told what the previous pages established as the running context
(vendor name, current section). Use that context — it often tells you the
gender / department even when the current page does not repeat it.

List EVERY distinct product visible on this page. A "product" is one unique
SKU / style code. If the same style is shown in N colorways with N codes,
return N rows.

Pay extra attention to:
  - Odd / off-pattern SKUs — multi-brand catalogs put different formats
    on the same page (e.g. ANODYNE W055-11-SHOE next to PROPET-prefixed codes).
  - SKUs in unusual positions — captions under thumbnails, small print
    in spec tables, alternate codes (e.g. global + USA item codes both
    printed for the same product).
  - Do NOT skip a SKU just because it doesn't match the prevailing format.

For each product return these fields. Use null when the field is not visible
on the page — do not invent. Preserve exact casing and punctuation as printed.

  sku           : the unique product identifier as printed
  description   : model / silhouette name as written by the vendor
                  (e.g. "SUPERSTAR VINTAGE", "Wave Creation 8")
  color         : color description as printed (e.g. "core black/core white",
                  "WHITE/BLACK-STADIUM GREEN")
  cost          : wholesale / cost price WITH currency + label as printed
                  (e.g. "WHSL $65.00", "110.0 USD"). null if only retail is shown.
  retail        : MSRP / retail price WITH currency + label as printed
                  (e.g. "MSRP $130", "RRP: $220"). If only one price is on
                  the page with no label, put it in retail.
  intro_date    : release / availability / intro / drop / ship date as printed
                  (e.g. "5/1/25", "JUL 2025", "2026-07-01")
  gender_hint  : any gender signal you can read — section header on the page,
                 caption like "WOMEN'S", SKU suffix/prefix like "W " or "WMNS"
                 or "GS". Quote it verbatim. null if nothing on the page hints.

Return ONLY JSON, no prose, no code fences:
{
  "products": [
    {"sku": "...", "description": "...", "color": "...", "cost": "...",
     "retail": "...", "intro_date": "...", "gender_hint": "..."},
    ...
  ]
}

If you see no products at all, return {"products": []}."""


YOLO_SETTINGS_SYSTEM = """You classify how dense a catalog page is with shoe images.

You see ONE page from a wholesale shoe catalog. Pick ONE density label:

  - "low"    : 1-6 shoes on the page. Includes hero/lifestyle pages with
               a couple of feature shots, or sparse pages with a small
               number of clearly-separated product photos.
  - "medium" : roughly 7-15 shoes in a moderate grid. Each shoe is a
               clear product photo but the page is starting to feel
               populated.
  - "high"   : 16+ shoes packed into a dense grid. Each individual
               shoe image is small (about 1/20 page width or less).

Examples:
  - Hero shot of a single shoe plus 2-3 color variants on the right
    → "low"
  - A 3x3 or 4x3 grid of product photos                  → "medium"
  - A 6-column grid of tiny thumbnails across the page  → "high"

When borderline (between low/medium, or medium/high), pick the HIGHER
density — over-detecting is recoverable, missing shoes is not.

Return ONLY JSON, no prose, no code fences:
{
  "density": "low" | "medium" | "high",
  "reasoning": "<short, ≤12 words>"
}"""

# Map density label → YOLO imgsz. Bias toward overshoot since the
# matcher absorbs extra detections semantically, but missing shoes
# (under-detection) is unrecoverable.
_DENSITY_TO_IMGSZ = {
    "low":    1280,
    "medium": 1920,
    "high":   2560,
}


IMAGE_MATCH_SYSTEM = """You match SKUs to numbered shoe images on a catalog page.

You will see an image of a single catalog page with RED numbered boxes
(1, 2, 3, ...) drawn around every detected shoe region. The numbers were
placed by a separate detector — they are NOT in the original page.

Some numbered boxes are per-SKU product photos (the catalog assigns a
SKU/style code to that shot). Others are hero / marketing / lifestyle
shots — large feature images, models wearing the shoe, context photos
— that don't carry a SKU.

You will also be given the list of SKUs that were extracted from the
page's TEXT. For each SKU, identify the numbered box that contains its
product image.

How to decide:
  - The SKU's product photo is usually the cleanest, most isolated shot.
    It tends to sit NEAR its SKU code, color name, and price in the page
    layout — small captions directly under or next to the image.
  - Hero / marketing / lifestyle shots tend to be larger, often on the
    LEFT or TOP of the page, without a SKU code printed near them.
  - When a SKU's image is not visible on the page (or wasn't detected
    by the box drawer), return null.
  - Never assign the same box to two different SKUs. If two SKUs seem
    to share a photo, pick the better match and null the other.

Return ONLY JSON, no prose, no code fences:
{
  "assignments": [
    {"sku": "<SKU string verbatim>", "box": <integer or null>},
    ...
  ]
}

Echo every SKU verbatim — do not change casing, whitespace, or
punctuation. Cover every SKU in the input exactly once."""


SKU_VALIDATE_SYSTEM = """You are a strict text-only validator. You will be given a list of strings extracted as SKUs from a wholesale shoe buy-sheet PDF. For EACH string, decide whether it actually looks like a product SKU / style code, or whether it's a false positive (something else that got mis-extracted into the SKU column).

A real SKU / style code typically looks like one of:
  - Short alphanumeric, often with a prefix and/or suffix
    (e.g. KJ2401, HM8850/100, W055-11-SHOE, D1GA270B01)
  - All-digit codes near product images (e.g. 9317-BLK-BOOT, 590196.0038)
  - Vendor-specific patterns (e.g. multi-part codes with dots, hyphens, slashes)

NOT a SKU — flag these as is_sku=false:
  - Prices ($120, 110.0 USD, RRP, MSRP, WHSL)
  - Sizes (US 9, M EUR, 7-15, OS)
  - Dates / season tags (FW26, SS25, 5/1/25, JUL 2025)
  - Page numbers (a bare 1, 2, 3 in the corner of the page)
  - Section headers / department names (MEN'S RUNNING, PREMIUM RANGE,
    WOMEN'S BOOTS)
  - Product NAMES that aren't codes (SUPERSTAR VINTAGE, AIR MAX 90)
  - Generic descriptive text (CORE, NEW, LATE ADD)

Borderline cases: if a string is shorter than 4 characters AND has no
digits, treat as suspicious (likely not a SKU). If it's a single word in
a closed-vocabulary slot (BLACK, WHITE), it's not a SKU.

Return ONLY JSON, no prose, no code fences:
{
  "results": [
    {"sku": "<string verbatim>", "is_sku": true},
    {"sku": "<string verbatim>", "is_sku": false, "reason": "<short, ≤5 words>"},
    ...
  ]
}

Cover every string exactly once. Echo each string verbatim — do not
canonicalize, normalize whitespace, or change casing."""


def image_to_b64_block(path: Path) -> dict:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type_for(path),
            "data": base64.standard_b64encode(path.read_bytes()).decode(),
        },
    }


def parse_llm_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("{"):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    s, e = raw.find("{"), raw.rfind("}")
    if s != -1 and e > s:
        return json.loads(raw[s:e + 1])
    raise json.JSONDecodeError("no JSON object found", raw, 0)


def context_block(prev_context: dict) -> str:
    vendor = prev_context.get("vendor") or "unknown"
    section = prev_context.get("current_section") or "unspecified"
    return (f"Running context carried from previous pages:\n"
            f"  vendor: {vendor}\n"
            f"  current section: {section}")


def extract_page(client: anthropic.Anthropic, model: str, image_path: Path,
                 prev_context: dict) -> tuple[list[dict], dict]:
    """Stage 1 — VLM full extract. Returns (candidates, usage). Candidates
    have all fields (sku/description/color/cost/retail/intro_date/gender_hint).
    These are CANDIDATES; the text-only validator in stage 2 filters them."""
    resp = client.messages.create(
        model=model,
        max_tokens=8192,
        system=EXTRACT_SYSTEM,
        messages=[{
            "role": "user",
            "content": [
                image_to_b64_block(image_path),
                {"type": "text", "text": context_block(prev_context)},
                {"type": "text", "text": "Extract every product on this page. Return JSON only."},
            ],
        }],
    )
    usage = usage_from_response(resp, model)
    parsed = parse_llm_json(resp.content[0].text)
    products = parsed.get("products", [])
    if not isinstance(products, list):
        return [], usage
    cleaned: list[dict] = []
    for p in products:
        if not isinstance(p, dict):
            continue
        cleaned.append({
            "sku": p.get("sku"),
            "description": p.get("description"),
            "color": p.get("color"),
            "cost": p.get("cost"),
            "retail": p.get("retail"),
            "intro_date": p.get("intro_date"),
            "gender_hint": p.get("gender_hint"),
        })
    return cleaned, usage


def validate_skus(client: anthropic.Anthropic, candidates: list[dict]
                  ) -> tuple[list[dict], list[dict], dict | None]:
    """Stage 2 — text-only LLM. Given the candidate products from stage 1,
    look ONLY at the SKU strings and decide which look like real SKUs.

    Returns (kept, rejected, usage) where:
      kept     — products whose SKU passed validation (unchanged shape)
      rejected — products whose SKU was flagged as not-a-SKU, augmented
                 with a `reason` string for debugging
      usage    — token usage dict (or None when no API call was made,
                 e.g. all-empty SKU list)

    Single Sonnet call, no image. Cheap (~$0.005 / page).
    """
    sku_strings = [str(c.get("sku") or "") for c in candidates]
    if not any(sku_strings):
        return [], [], None
    body = "Strings to validate:\n" + "\n".join(
        f"  - {s if s else '<empty>'}" for s in sku_strings
    )
    resp = client.messages.create(
        model=MODEL_SONNET,
        max_tokens=4096,
        system=SKU_VALIDATE_SYSTEM,
        messages=[{"role": "user", "content": body}],
    )
    usage = usage_from_response(resp, MODEL_SONNET)
    try:
        parsed = parse_llm_json(resp.content[0].text)
    except json.JSONDecodeError:
        # Validator failed — fall back to keeping everything (don't drop
        # extracted products on a validator hiccup).
        return list(candidates), [], usage
    raw = parsed.get("results", [])
    if not isinstance(raw, list):
        return list(candidates), [], usage
    # Map verdict by sku string. The validator should echo verbatim, but
    # build a forgiving lookup anyway.
    verdict: dict[str, dict] = {}
    for r in raw:
        if not isinstance(r, dict):
            continue
        s = r.get("sku")
        if isinstance(s, str):
            verdict[s.strip()] = r
    kept: list[dict] = []
    rejected: list[dict] = []
    for c in candidates:
        s = str(c.get("sku") or "").strip()
        v = verdict.get(s)
        if v is None or v.get("is_sku") is True:
            # Missing verdict → default keep (don't drop on validator gaps).
            kept.append(c)
        else:
            entry = dict(c)
            entry["reason"] = v.get("reason") or "rejected by sku validator"
            rejected.append(entry)
    return kept, rejected, usage


_pdf_lock = threading.Lock()  # PyMuPDF is not thread-safe; serialize renders


def pick_yolo_settings(client: anthropic.Anthropic, page_image_path: Path
                       ) -> tuple[int, dict | None, str | None]:
    """One Sonnet vision call: classify the page's shoe-image density
    (low / medium / high) and return the YOLO imgsz that fits.

    Returns (imgsz, usage, reasoning). On any failure falls back to the
    "medium" mapping (1920) — covers most catalog layouts safely.
    """
    fallback_density = "medium"
    fallback_imgsz = _DENSITY_TO_IMGSZ[fallback_density]
    try:
        resp = client.messages.create(
            model=MODEL_SONNET,
            max_tokens=256,
            system=YOLO_SETTINGS_SYSTEM,
            messages=[{
                "role": "user",
                "content": [
                    image_to_b64_block(page_image_path),
                    {"type": "text",
                     "text": "Classify the density of this catalog page. JSON only."},
                ],
            }],
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  picker: failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return fallback_imgsz, None, fallback_density
    usage = usage_from_response(resp, MODEL_SONNET)
    try:
        parsed = parse_llm_json(resp.content[0].text)
    except json.JSONDecodeError:
        return fallback_imgsz, usage, fallback_density
    density = (parsed.get("density") or "").strip().lower()
    if density not in _DENSITY_TO_IMGSZ:
        return fallback_imgsz, usage, fallback_density
    reasoning = parsed.get("reasoning") or ""
    label = f"{density}: {reasoning}" if reasoning else density
    return _DENSITY_TO_IMGSZ[density], usage, label


def match_images_to_skus(client: anthropic.Anthropic, products: list[dict],
                         annotated_image_path: Path, n_boxes: int
                         ) -> tuple[dict[str, int], dict | None]:
    """Sonnet vision call: send the annotated page (with numbered red
    boxes drawn by detect.shoe.annotate_page) plus the extracted SKU list
    and ask which numbered box belongs to each SKU.

    Returns ({sku_string: box_number}, usage_dict). Missing SKUs (the
    matcher said null or omitted) are absent from the dict. Box numbers
    are 1-indexed and bounded to 1..n_boxes; out-of-range values are
    treated as null. Two SKUs claiming the same box are de-duped (the
    first SKU wins, the second falls through to no image).
    """
    if not products or n_boxes <= 0:
        return {}, None
    body_lines = [f"Annotated page image (boxes 1..{n_boxes} drawn in red).",
                  "Extracted SKUs from the page text:"]
    for i, p in enumerate(products, start=1):
        sku = (p.get("sku") or "").strip()
        desc = (p.get("description") or "").strip()
        color = (p.get("color") or "").strip()
        body_lines.append(
            f"  {i}. sku={sku!r}  desc={desc!r}  color={color!r}"
        )
    body_lines.append("")
    body_lines.append(
        "For each SKU return the matching box number (1.."
        f"{n_boxes}) or null. Echo each sku verbatim."
    )
    body = "\n".join(body_lines)
    resp = client.messages.create(
        model=MODEL_SONNET,
        max_tokens=2048,
        system=IMAGE_MATCH_SYSTEM,
        messages=[{
            "role": "user",
            "content": [
                image_to_b64_block(annotated_image_path),
                {"type": "text", "text": body},
            ],
        }],
    )
    usage = usage_from_response(resp, MODEL_SONNET)
    try:
        parsed = parse_llm_json(resp.content[0].text)
    except json.JSONDecodeError:
        return {}, usage
    raw = parsed.get("assignments", [])
    if not isinstance(raw, list):
        return {}, usage
    used_boxes: set[int] = set()
    mapping: dict[str, int] = {}
    for a in raw:
        if not isinstance(a, dict):
            continue
        sku = a.get("sku")
        box = a.get("box")
        if not isinstance(sku, str):
            continue
        if not isinstance(box, int):
            continue
        if box < 1 or box > n_boxes:
            continue
        if box in used_boxes:
            continue  # first-SKU-wins de-dup
        mapping[sku.strip()] = box
        used_boxes.add(box)
    return mapping, usage


def assign_images_via_match(client: anthropic.Anthropic, products: list[dict],
                            page_info: dict | None
                            ) -> tuple[str | None, dict | None]:
    """Set `image_path` on each product by asking the VLM which numbered
    box on the annotated page belongs to which SKU.

    `page_info` is the per-page detection entry produced by
    detect.shoe.run_detect_prepass: {"crops":[paths], "bboxes":[...],
    "annotated": path}. None or missing entries mean detection didn't
    run for this page (multi-doc mode, no weights, etc.) — every product
    gets image_path=None silently.

    Returns (marker, usage). Marker is one of:
      None              — at least one product got an image (or detection
                          was skipped entirely; that's silent on purpose)
      "no_match"        — matcher returned nothing useful; products kept
                          but no images attached. Surfaces in REVIEW.
      "matcher_failed"  — matcher call raised; same effect as no_match.
    """
    for p in products:
        p["image_path"] = None
    if not products:
        return None, None
    if not page_info:
        return None, None
    crops = page_info.get("crops") or []
    annotated = page_info.get("annotated")
    if not crops or not annotated:
        return None, None
    annotated_path = Path(annotated)
    if not annotated_path.exists():
        return None, None
    try:
        mapping, usage = match_images_to_skus(
            client, products, annotated_path, len(crops),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  matcher: failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return "matcher_failed", None
    n_attached = 0
    for p in products:
        sku_raw = (p.get("sku") or "")
        box = mapping.get(sku_raw.strip())
        if box is None:
            continue
        slot = box - 1
        if 0 <= slot < len(crops):
            p["image_path"] = crops[slot]
            n_attached += 1
    if n_attached == 0:
        return "no_match", usage
    return None, usage


def ensure_page_image(pdf: fitz.Document, page_no: int,
                      pages_dir: Path) -> Path:
    """Return the path to a rendered page image — cached if present,
    freshly rendered if not. Adaptive: PNG first, JPEG fallback for
    photo-heavy pages PNG can't compress. Raises ImageTooLargeError if
    no encoder strategy fits."""
    cached = find_cached_render(pages_dir, page_no)
    if cached is not None:
        return cached
    with _pdf_lock:
        return render_page_safe(pdf, page_no, pages_dir)


def process_page(client: anthropic.Anthropic, model: str,
                 pdf: fitz.Document, page_no: int,
                 pages_dir: Path, force: bool,
                 crops_by_page: dict[int, dict] | None = None) -> dict:
    """Returns a dict with {page_no, status, n_products, error?} for the summary."""
    json_path = pages_dir / f"{page_no:02d}.json"
    if not json_path.exists():
        return {"page_no": page_no, "status": "skip",
                "reason": "no classify json", "n_products": 0}
    record = json.loads(json_path.read_text())
    if record.get("label") != "product":
        return {"page_no": page_no, "status": "skip",
                "reason": f"label={record.get('label')}", "n_products": 0}
    if "products" in record and not force:
        n = len(record["products"])
        return {"page_no": page_no, "status": "cached", "n_products": n}
    page_info = (crops_by_page or {}).get(page_no)

    try:
        image_path = ensure_page_image(pdf, page_no, pages_dir)
    except ImageTooLargeError as exc:
        record["error"] = f"image_too_large: {exc}"
        record.pop("products", None)
        record.pop("sku_hint", None)
        json_path.write_text(json.dumps(record, indent=2))
        return {"page_no": page_no, "status": "error",
                "error": str(exc), "n_products": 0}

    prev_context = record.get("prev_context") or {}

    # Stage 1 — VLM full extract (candidate products with all fields).
    try:
        candidates, extract_usage = extract_page(client, model, image_path, prev_context)
    except Exception as exc:  # noqa: BLE001
        record["error"] = f"extract_failed: {type(exc).__name__}: {exc}"
        json_path.write_text(json.dumps(record, indent=2))
        return {"page_no": page_no, "status": "error",
                "error": str(exc), "n_products": 0}

    record["n_candidates"] = len(candidates)
    existing_usage = record.get("usage") or {}
    existing_usage["extract"] = extract_usage
    record["usage"] = existing_usage

    # Deterministic text-layer verification — substring-match every
    # VLM-emitted field against the PDF text layer. text_layer_present
    # is False when PyMuPDF raises or the canon'd text is empty; in that
    # case we fall through to today's Stage 2 LLM validator.
    kept, dropped, text_layer_present, text_layer_error = verify_against_text_layer(
        candidates, pdf, page_no, _pdf_lock,
    )
    record["text_layer_present"] = text_layer_present
    if text_layer_error:
        record["text_layer_error"] = text_layer_error
    else:
        record.pop("text_layer_error", None)

    if text_layer_present:
        record.pop("error", None)
        record.pop("sku_hint", None)       # legacy field
        record.pop("candidates", None)     # legacy field from prior refactor
        record["products"] = kept
        record["rejected_candidates"] = dropped
        # No Stage 2 call on the deterministic path.
        record["usage"]["validate"] = None
        marker, match_usage = assign_images_via_match(client, kept, page_info)
        record["usage"]["match"] = match_usage
        if marker:
            record["image_association"] = marker
        else:
            record.pop("image_association", None)
        json_path.write_text(json.dumps(record, indent=2))
        return {"page_no": page_no, "status": "ok",
                "n_products": len(kept), "n_candidates": len(candidates),
                "n_rejected": len(dropped), "path": "deterministic"}

    # Stage 2 fallback — text-only SKU validator (filters false-positive SKUs).
    try:
        products, rejected, validate_usage = validate_skus(client, candidates)
    except Exception as exc:  # noqa: BLE001
        # Don't lose stage 1 work on a validator hiccup — keep all candidates.
        record["error"] = f"validate_skus_failed: {type(exc).__name__}: {exc}"
        record["products"] = candidates
        record["rejected_candidates"] = []
        json_path.write_text(json.dumps(record, indent=2))
        return {"page_no": page_no, "status": "error",
                "error": str(exc), "n_products": len(candidates)}

    for r in rejected:
        r["stage"] = "llm_validator"

    record.pop("error", None)
    record.pop("sku_hint", None)       # legacy field
    record.pop("candidates", None)     # legacy field from prior refactor
    record["products"] = products
    record["rejected_candidates"] = rejected
    record["usage"]["validate"] = validate_usage
    marker, match_usage = assign_images_via_match(client, products, page_info)
    record["usage"]["match"] = match_usage
    if marker:
        record["image_association"] = marker
    else:
        record.pop("image_association", None)
    json_path.write_text(json.dumps(record, indent=2))
    return {"page_no": page_no, "status": "ok",
            "n_products": len(products), "n_candidates": len(candidates),
            "n_rejected": len(rejected), "path": "llm_validator"}


def _detect_prepass(client: anthropic.Anthropic, pdf: fitz.Document,
                    pages: list[int], pages_dir: Path,
                    *, force: bool, detect_images: bool
                    ) -> dict[int, dict]:
    """Run the serial footwear-detection pre-pass on product pages.

    Returns {} when detection is disabled (multi-doc runs, missing
    dependency, missing weights). Otherwise returns
    {page_no: {"crops": [...], "bboxes": [...], "annotated": path}}.
    Page images are rendered/cached via ensure_page_image so the existing
    .pages/NN.{png,jpg} files are reused.

    Per-page YOLO imgsz is picked by a Sonnet vision probe so dense
    grids (adidas: ~30 tiny shoes / page) and hero-shot layouts
    (Converse: 1-2 large shoes / page) each get an appropriately scaled
    inference resolution. The pick is cached in manifest.json so re-runs
    don't repeat the call.
    """
    if not detect_images:
        return {}
    if not shoe_detect.is_available():
        return {}
    targets: list[tuple[int, Path]] = []
    for p in pages:
        json_path = pages_dir / f"{p:02d}.json"
        if not json_path.exists():
            continue
        try:
            rec = json.loads(json_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if rec.get("label") != "product":
            continue
        try:
            image_path = ensure_page_image(pdf, p, pages_dir)
        except ImageTooLargeError:
            continue
        targets.append((p, image_path))
    if not targets:
        return {}
    print(f"detecting footwear on {len(targets)} product pages (serial pass)",
          file=sys.stderr)

    def _picker(page_no: int, image_path: Path
                ) -> tuple[int, dict | None, str | None]:
        return pick_yolo_settings(client, image_path)

    return shoe_detect.run_detect_prepass(
        targets, pages_dir, force=force, settings_picker=_picker,
    )


def _backfill_image_paths(client: anthropic.Anthropic, pages_dir: Path,
                          crops_by_page: dict[int, dict]) -> int:
    """Run the matcher on cached page JSONs whose products were extracted
    before detection had run. Returns the number of pages mutated. No-op
    when crops_by_page is empty.

    Only touches pages where (a) the cached JSON has products and (b) no
    product already has an image_path set — leaves prior runs alone.
    """
    if not crops_by_page:
        return 0
    n = 0
    for page_no, page_info in crops_by_page.items():
        crops = (page_info or {}).get("crops") or []
        if not crops:
            continue
        json_path = pages_dir / f"{page_no:02d}.json"
        if not json_path.exists():
            continue
        try:
            rec = json.loads(json_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        products = rec.get("products")
        if not products:
            continue
        already_set = any(p.get("image_path") for p in products)
        if already_set:
            continue
        marker, match_usage = assign_images_via_match(
            client, products, page_info,
        )
        rec.setdefault("usage", {})["match"] = match_usage
        if marker:
            rec["image_association"] = marker
        else:
            rec.pop("image_association", None)
        json_path.write_text(json.dumps(rec, indent=2))
        n += 1
    if n:
        print(f"backfilled image_path on {n} cached page(s) via matcher",
              file=sys.stderr)
    return n


def extract_pdf(pdf_path: Path, only_page: int | None, force: bool,
                workers: int, model: str,
                detect_images: bool = False) -> None:
    pages_dir = pdf_path.with_name(pdf_path.stem + ".pages")
    if not pages_dir.exists():
        print(f"error: {pages_dir} not found — run classify_pages.py first",
              file=sys.stderr)
        sys.exit(1)
    pdf = fitz.open(str(pdf_path))
    if only_page is not None:
        if not (1 <= only_page <= pdf.page_count):
            raise ValueError(f"--page {only_page} out of range 1..{pdf.page_count}")
        pages = [only_page]
    else:
        pages = list(range(1, pdf.page_count + 1))

    client = anthropic.Anthropic(max_retries=10)
    results: list[dict] = []

    try:
        crops_by_page = _detect_prepass(
            client, pdf, pages, pages_dir, force=force,
            detect_images=detect_images,
        )
        _backfill_image_paths(client, pages_dir, crops_by_page)

        print(f"extracting {len(pages)} pages with {workers} workers (model={model})",
              file=sys.stderr)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            fut_to_page = {
                pool.submit(process_page, client, model, pdf, p,
                            pages_dir, force, crops_by_page): p
                for p in pages
            }
            for fut in as_completed(fut_to_page):
                page_no = fut_to_page[fut]
                try:
                    res = fut.result()
                except Exception as exc:  # noqa: BLE001
                    res = {"page_no": page_no, "status": "error",
                           "error": f"worker crashed: {exc}", "n_products": 0}
                results.append(res)
                _log_page_result(res)
    finally:
        pdf.close()

    _print_summary(results, pages_dir)


def _log_page_result(res: dict) -> None:
    p = res["page_no"]
    s = res["status"]
    if s == "ok":
        n_p = res["n_products"]
        n_c = res.get("n_candidates", 0)
        n_r = res.get("n_rejected", 0)
        path = res.get("path", "?")
        print(f"  page {p:02d}: {n_p} products ({n_c} candidates, "
              f"{n_r} rejected, path={path})", file=sys.stderr)
    elif s == "cached":
        print(f"  page {p:02d}: cached ({res['n_products']} products)", file=sys.stderr)
    elif s == "skip":
        print(f"  page {p:02d}: skip ({res.get('reason','')})", file=sys.stderr)
    elif s == "error":
        print(f"  page {p:02d}: ERROR {res.get('error','')}", file=sys.stderr)


def _print_summary(results: list[dict], pages_dir: Path) -> None:
    n_ok = sum(1 for r in results if r["status"] in ("ok", "cached"))
    n_skip = sum(1 for r in results if r["status"] == "skip")
    n_err = sum(1 for r in results if r["status"] == "error")
    n_products = sum(r["n_products"] for r in results)
    print("", file=sys.stderr)
    print(f"SUMMARY: {len(results)} pages — "
          f"{n_ok} extracted, {n_skip} skipped, {n_err} FAILED — "
          f"{n_products} products total -> {pages_dir}", file=sys.stderr)
    if n_err:
        print("Failed pages:", file=sys.stderr)
        for r in sorted([r for r in results if r["status"] == "error"],
                        key=lambda r: r["page_no"]):
            print(f"  page {r['page_no']:02d}: {r.get('error','')}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--force", action="store_true",
                    help="Re-extract pages even if products exist")
    ap.add_argument("--page", type=int, default=None,
                    help="Only process this 1-indexed page")
    ap.add_argument("--workers", type=int, default=5,
                    help="Number of parallel extraction workers (default 5)")
    ap.add_argument("--model", choices=("opus", "sonnet"), default="opus",
                    help="Which model to use for extract (default opus)")
    ap.add_argument("--detect-images", action="store_true",
                    help="Run YOLO footwear detection + per-product crops "
                         "(single-doc only; needs ultralytics + weights)")
    args = ap.parse_args()
    if not args.pdf.exists():
        ap.error(f"PDF not found: {args.pdf}")
    model = MODEL_OPUS if args.model == "opus" else MODEL_SONNET
    extract_pdf(args.pdf, args.page, args.force, args.workers, model,
                detect_images=args.detect_images)


if __name__ == "__main__":
    main()
