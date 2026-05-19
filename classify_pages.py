"""Step 1 of agentic per-page parsing: classify each PDF page and track a
running context window.

Per page (atom of work):
  1. Render the page to <doc>.pages/<NN>.png with the safe size guard
     (downscales 75% per pass up to 4 passes to stay under Anthropic's
     5 MiB base64 cap). If even after 4 passes it's too big, the page
     is marked with `error: "image_too_large: ..."` and label="unknown".
  2. Send to Claude Sonnet vision; get {label, ...metadata}.
  3. Apply the deterministic context update (vendor / current_section).
  4. Write <doc>.pages/<NN>.json with {page_no, label, ...metadata,
     prev_context, context_after, error?}.

The classifier sees only the page image — no prior context — so its LLM
calls are independent. They run in parallel via ThreadPoolExecutor. The
context fold (prev_context = previous page's context_after) is sequential
and runs after all classifications complete.

Per-page outputs live in files/<vendor>/<doc>.pages/<NN>.{png,json}. The
running context propagates via prev_context / context_after fields, so a
crash after page N leaves <N>.json intact and reruns skip what's done.

Usage:
  python agentic_parsing/classify_pages.py files/<vendor>/<doc>.pdf
  python agentic_parsing/classify_pages.py files/<vendor>/<doc>.pdf --workers 8
  python agentic_parsing/classify_pages.py files/<vendor>/<doc>.pdf --force
  python agentic_parsing/classify_pages.py files/<vendor>/<doc>.pdf --page 12
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import threading
from collections import Counter
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

load_dotenv()

MODEL = "claude-sonnet-4-6"

LABELS = ("category", "brand_name", "index_or_other", "product", "unknown")

SYSTEM = """You classify a single page of a wholesale buy-sheet / vendor line-sheet PDF.

Pick EXACTLY ONE label that best describes the page:

- "category"        Section divider whose dominant content is a category /
                    department name ("FOOTWEAR", "MEN'S APPAREL",
                    "ACCESSORIES", "FW25", etc.). Big text, usually no
                    products, sometimes a single hero image.
- "brand_name"      Brand cover / splash page. Dominated by the brand logo
                    or name, possibly a season tagline. No product detail.
- "index_or_other"  Table of contents, legal / copyright, lifestyle or
                    lookbook imagery with no extractable product info. NOT
                    a section divider.
- "product"         Page with actual extractable product content: SKUs,
                    prices, descriptions, colorways, swatches, line-sheet
                    tables, product grids, single-product hero specs, etc.
- "unknown"         None of the above clearly fits.

If the page is hybrid (e.g. a section opener with a few products), pick the
DOMINANT label — do not invent new labels.

Required output fields by label:
  category   -> {"label":"category","name":"<section name as written>"}
  brand_name -> {"label":"brand_name","brand":"<brand as written>"}
  any other  -> {"label":"<label>"}

Return EXACTLY one JSON object. No prose, no code fences."""


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


def classify_page(client: anthropic.Anthropic, image_path: Path) -> tuple[dict, dict]:
    resp = client.messages.create(
        model=MODEL,
        max_tokens=256,
        system=SYSTEM,
        messages=[{
            "role": "user",
            "content": [
                image_to_b64_block(image_path),
                {"type": "text", "text": "Classify this page. Return the JSON."},
            ],
        }],
    )
    parsed = parse_llm_json(resp.content[0].text)
    label = parsed.get("label", "unknown")
    if label not in LABELS:
        label = "unknown"
    out = {"label": label}
    if label == "category":
        name = parsed.get("name")
        if isinstance(name, str) and name.strip():
            out["name"] = name.strip()
    elif label == "brand_name":
        brand = parsed.get("brand")
        if isinstance(brand, str) and brand.strip():
            out["brand"] = brand.strip()
    return out, usage_from_response(resp, MODEL)


def apply_context_update(prev: dict, page_info: dict) -> dict:
    ctx = dict(prev)
    if page_info.get("label") == "category" and "name" in page_info:
        ctx["current_section"] = page_info["name"]
    elif page_info.get("label") == "brand_name" and "brand" in page_info:
        ctx["vendor"] = page_info["brand"]
    return ctx


_pdf_lock = threading.Lock()


def classify_one_page(client: anthropic.Anthropic, pdf: fitz.Document,
                      page_no: int, pages_dir: Path, force: bool) -> dict:
    """Render + classify ONE page. Returns:
      {"page_no": N, "page_info": {label, ...}, "error": str|None}
    Skips the API call if a JSON record already exists and --force is off."""
    json_path = pages_dir / f"{page_no:02d}.json"
    if json_path.exists() and not force:
        existing = json.loads(json_path.read_text())
        # Recover the page_info needed for context fold without an API call
        page_info = {"label": existing.get("label", "unknown")}
        for k in ("name", "brand"):
            if k in existing:
                page_info[k] = existing[k]
        return {"page_no": page_no, "page_info": page_info,
                "error": existing.get("error"), "cached": True}
    image_path = find_cached_render(pages_dir, page_no)
    if image_path is None:
        try:
            with _pdf_lock:
                image_path = render_page_safe(pdf, page_no, pages_dir)
        except ImageTooLargeError as exc:
            return {"page_no": page_no,
                    "page_info": {"label": "unknown"},
                    "error": f"image_too_large: {exc}", "cached": False}
    try:
        page_info, usage = classify_page(client, image_path)
        return {"page_no": page_no, "page_info": page_info,
                "error": None, "cached": False, "usage": usage}
    except Exception as exc:  # noqa: BLE001
        return {"page_no": page_no,
                "page_info": {"label": "unknown"},
                "error": f"{type(exc).__name__}: {exc}", "cached": False}


def _write_page_json(pages_dir: Path, page_no: int, page_info: dict,
                     prev_context: dict, context_after: dict,
                     error: str | None, usage: dict | None = None) -> None:
    record = {
        "page_no": page_no,
        **page_info,
        "prev_context": prev_context,
        "context_after": context_after,
    }
    if error:
        record["error"] = error
    if usage:
        record["usage"] = {"classify": usage}
    (pages_dir / f"{page_no:02d}.json").write_text(json.dumps(record, indent=2))


def _read_prev_context(pages_dir: Path, page_no: int) -> dict:
    empty = {"vendor": None, "current_section": None}
    if page_no <= 1:
        return empty
    prev_json = pages_dir / f"{page_no - 1:02d}.json"
    if not prev_json.exists():
        return empty
    return json.loads(prev_json.read_text()).get("context_after", empty)


def fold_and_write(results: list[dict], pages_dir: Path) -> None:
    """Walk all classified pages in order, computing prev_context /
    context_after and writing the per-page JSON. Skips writes for cached
    pages (their JSON already has the correct context from a prior run)."""
    prev_context = {"vendor": None, "current_section": None}
    for r in sorted(results, key=lambda r: r["page_no"]):
        context_after = apply_context_update(prev_context, r["page_info"])
        if not r["cached"]:
            _write_page_json(pages_dir, r["page_no"], r["page_info"],
                             prev_context, context_after, r["error"],
                             usage=r.get("usage"))
        prev_context = context_after


def _log_page_result(r: dict) -> None:
    p = r["page_no"]
    info = r["page_info"]
    err = r["error"]
    extra = info.get("name") or info.get("brand") or ""
    suffix = f"  ({extra})" if extra else ""
    if err:
        print(f"  page {p:02d}: ERROR {err}", file=sys.stderr)
    elif r["cached"]:
        print(f"  page {p:02d}: cached label={info['label']}{suffix}", file=sys.stderr)
    else:
        print(f"  page {p:02d}: {info['label']}{suffix}", file=sys.stderr)


def _print_summary(results: list[dict], pages_dir: Path) -> None:
    labels = Counter(r["page_info"]["label"] for r in results if not r["error"])
    failed = [r for r in results if r["error"]]
    summary = ", ".join(f"{n} {lab}" for lab, n in labels.most_common())
    print("", file=sys.stderr)
    print(f"SUMMARY: {len(results)} pages -> {pages_dir}  [{summary}"
          f"{', ' + str(len(failed)) + ' FAILED' if failed else ''}]",
          file=sys.stderr)
    if failed:
        print("Failed pages:", file=sys.stderr)
        for r in sorted(failed, key=lambda r: r["page_no"]):
            print(f"  page {r['page_no']:02d}: {r['error']}", file=sys.stderr)


def classify_pdf(pdf_path: Path, only_page: int | None, force: bool,
                 workers: int) -> None:
    pages_dir = pdf_path.with_name(pdf_path.stem + ".pages")
    pages_dir.mkdir(exist_ok=True)
    pdf = fitz.open(str(pdf_path))
    if only_page is not None:
        if not (1 <= only_page <= pdf.page_count):
            raise ValueError(f"--page {only_page} out of range 1..{pdf.page_count}")
        pages = [only_page]
    else:
        pages = list(range(1, pdf.page_count + 1))
    client = anthropic.Anthropic(max_retries=10)
    print(f"classifying {len(pages)} pages with {workers} workers",
          file=sys.stderr)
    results: list[dict] = []
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            fut_to_page = {
                pool.submit(classify_one_page, client, pdf, p, pages_dir, force): p
                for p in pages
            }
            for fut in as_completed(fut_to_page):
                r = fut.result()
                results.append(r)
                _log_page_result(r)
    finally:
        if only_page is None:
            fold_and_write(results, pages_dir)
        else:
            # Single-page mode: don't touch the rest of the doc. Pull
            # prev_context from the existing JSON of page N-1.
            r = results[0]
            if not r["cached"]:
                prev_context = _read_prev_context(pages_dir, r["page_no"])
                context_after = apply_context_update(prev_context, r["page_info"])
                _write_page_json(pages_dir, r["page_no"], r["page_info"],
                                 prev_context, context_after, r["error"],
                                 usage=r.get("usage"))
        pdf.close()
    _print_summary(results, pages_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=Path,
                    help="Vendor PDF, e.g. files/<vendor>/<doc>.pdf")
    ap.add_argument("--force", action="store_true",
                    help="Re-classify pages even if <NN>.json exists")
    ap.add_argument("--page", type=int, default=None,
                    help="Only process this 1-indexed page (skips context fold)")
    ap.add_argument("--workers", type=int, default=5,
                    help="Number of parallel classify workers (default 5)")
    args = ap.parse_args()
    if not args.pdf.exists():
        ap.error(f"PDF not found: {args.pdf}")
    classify_pdf(args.pdf, args.page, args.force, args.workers)


if __name__ == "__main__":
    main()
