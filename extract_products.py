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
from _render import ImageTooLargeError, render_page_png_safe  # noqa: E402

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

  sku          : the unique product identifier as printed
  description  : model / silhouette name as written by the vendor
                 (e.g. "SUPERSTAR VINTAGE", "Wave Creation 8")
  color        : color description as printed (e.g. "core black/core white",
                 "WHITE/BLACK-STADIUM GREEN")
  cost         : wholesale / cost price WITH currency + label as printed
                 (e.g. "WHSL $65.00", "110.0 USD"). null if only retail is shown.
  retail       : MSRP / retail price WITH currency + label as printed
                 (e.g. "MSRP $130", "RRP: $220"). If only one price is on
                 the page with no label, put it in retail.
  intro_date   : release / availability / intro / drop / ship date as printed
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


def png_to_b64_block(png_path: Path) -> dict:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": base64.standard_b64encode(png_path.read_bytes()).decode(),
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


def extract_page(client: anthropic.Anthropic, model: str, png_path: Path,
                 prev_context: dict) -> list[dict]:
    """Stage 1 — VLM full extract. Returns the list of candidate products
    with all fields (sku/description/color/cost/retail/intro_date/gender_hint).
    These are CANDIDATES; the text-only validator in stage 2 filters them."""
    resp = client.messages.create(
        model=model,
        max_tokens=8192,
        system=EXTRACT_SYSTEM,
        messages=[{
            "role": "user",
            "content": [
                png_to_b64_block(png_path),
                {"type": "text", "text": context_block(prev_context)},
                {"type": "text", "text": "Extract every product on this page. Return JSON only."},
            ],
        }],
    )
    parsed = parse_llm_json(resp.content[0].text)
    products = parsed.get("products", [])
    if not isinstance(products, list):
        return []
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
    return cleaned


def validate_skus(client: anthropic.Anthropic, candidates: list[dict]
                  ) -> tuple[list[dict], list[dict]]:
    """Stage 2 — text-only LLM. Given the candidate products from stage 1,
    look ONLY at the SKU strings and decide which look like real SKUs.

    Returns (kept, rejected) where:
      kept     — products whose SKU passed validation (unchanged shape)
      rejected — products whose SKU was flagged as not-a-SKU, augmented
                 with a `reason` string for debugging

    Single Sonnet call, no image. Cheap (~$0.005 / page).
    """
    sku_strings = [str(c.get("sku") or "") for c in candidates]
    if not any(sku_strings):
        return [], []
    body = "Strings to validate:\n" + "\n".join(
        f"  - {s if s else '<empty>'}" for s in sku_strings
    )
    resp = client.messages.create(
        model=MODEL_SONNET,
        max_tokens=4096,
        system=SKU_VALIDATE_SYSTEM,
        messages=[{"role": "user", "content": body}],
    )
    try:
        parsed = parse_llm_json(resp.content[0].text)
    except json.JSONDecodeError:
        # Validator failed — fall back to keeping everything (don't drop
        # extracted products on a validator hiccup).
        return list(candidates), []
    raw = parsed.get("results", [])
    if not isinstance(raw, list):
        return list(candidates), []
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
    return kept, rejected


_pdf_lock = threading.Lock()  # PyMuPDF is not thread-safe; serialize renders


def ensure_png(pdf: fitz.Document, page_no: int, png_path: Path) -> None:
    """Render the PNG if it doesn't exist yet (or render failed earlier).
    Raises ImageTooLargeError on failure — caller marks the page."""
    if png_path.exists() and png_path.stat().st_size > 0:
        return
    with _pdf_lock:
        render_page_png_safe(pdf, page_no, png_path)


def process_page(client: anthropic.Anthropic, model: str,
                 pdf: fitz.Document, page_no: int,
                 pages_dir: Path, force: bool) -> dict:
    """Returns a dict with {page_no, status, n_products, error?} for the summary."""
    json_path = pages_dir / f"{page_no:02d}.json"
    png_path = pages_dir / f"{page_no:02d}.png"
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

    try:
        ensure_png(pdf, page_no, png_path)
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
        candidates = extract_page(client, model, png_path, prev_context)
    except Exception as exc:  # noqa: BLE001
        record["error"] = f"extract_failed: {type(exc).__name__}: {exc}"
        json_path.write_text(json.dumps(record, indent=2))
        return {"page_no": page_no, "status": "error",
                "error": str(exc), "n_products": 0}

    # Stage 2 — text-only SKU validator (filters false-positive SKUs).
    try:
        products, rejected = validate_skus(client, candidates)
    except Exception as exc:  # noqa: BLE001
        # Don't lose stage 1 work on a validator hiccup — keep all candidates.
        record["error"] = f"validate_skus_failed: {type(exc).__name__}: {exc}"
        record["products"] = candidates
        record["rejected_candidates"] = []
        json_path.write_text(json.dumps(record, indent=2))
        return {"page_no": page_no, "status": "error",
                "error": str(exc), "n_products": len(candidates)}

    record.pop("error", None)
    record.pop("sku_hint", None)       # legacy field
    record.pop("candidates", None)     # legacy field from prior refactor
    record["products"] = products
    record["rejected_candidates"] = rejected
    json_path.write_text(json.dumps(record, indent=2))
    return {"page_no": page_no, "status": "ok",
            "n_products": len(products), "n_candidates": len(candidates),
            "n_rejected": len(rejected)}


def extract_pdf(pdf_path: Path, only_page: int | None, force: bool,
                workers: int, model: str) -> None:
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
    print(f"extracting {len(pages)} pages with {workers} workers (model={model})",
          file=sys.stderr)
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            fut_to_page = {
                pool.submit(process_page, client, model, pdf, p, pages_dir, force): p
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
        print(f"  page {p:02d}: {n_p} products ({n_c} candidates, {n_r} rejected)",
              file=sys.stderr)
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
    args = ap.parse_args()
    if not args.pdf.exists():
        ap.error(f"PDF not found: {args.pdf}")
    model = MODEL_OPUS if args.model == "opus" else MODEL_SONNET
    extract_pdf(args.pdf, args.page, args.force, args.workers, model)


if __name__ == "__main__":
    main()
