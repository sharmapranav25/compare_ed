# SCOPE — kithbuysheet2

Scope, runtime requirements, verified behavior, and known gaps for the
agentic per-page buy-sheet pipeline. Use this alongside [CLAUDE.md](CLAUDE.md)
(which describes the pipeline mechanics) to know what this repo *is* and *is
not* responsible for.

## In scope

- Converting one vendor wholesale shoe buy-sheet / line-sheet PDF into one
  filled `BUYSHEET_<vendor>.xlsx`.
- Three-step agentic pipeline:
  1. [classify_pages.py](classify_pages.py) — Sonnet vision labels each
     page (`category` / `brand_name` / `product` / `index_or_other` /
     `unknown`) and folds a running `(vendor, current_section)` context.
  2. [extract_products.py](extract_products.py) — two-stage pass per
     `product` page: Opus visual extract (recall) → Sonnet text-only
     SKU validator (precision).
  3. [build_buysheet.py](build_buysheet.py) — aggregates per-page JSON into
     a copy of `BUYSHEET_template.xlsx`, resolving each dropdown cell via
     [vocab_map.py](vocab_map.py) with on-disk caching.
- Per-PDF on-disk state under `<pdf-stem>.pages/` (PNG + JSON per page).
  Crash-resumable; each step idempotent unless `--force`.
- Cross-document dropdown cache at [cache/](cache/) keyed by normalized
  primary signal (description / color / date).
- Failure surfacing: pages with `error: ...` and pages labelled `product`
  with 0 products extracted get yellow STYLE# tint plus a `REVIEW` sheet.

## Out of scope

- The legacy deterministic `phase1/` → `phase5/` pipeline lives in the
  original `5 phase kith sheet agent` repo; not present here.
- The `BUYSHEET_template.xlsx` itself — must be supplied externally (see
  Runtime setup).
- Multi-PDF batch runs, vendor de-dup across PDFs, or downstream upload to
  any system. One PDF in, one xlsx out.
- Any UI / web layer. CLI only.
- Schema migrations to the template. The pipeline reads the `Product Data`
  sheet at runtime for dropdown vocabs — if the template's dropdown
  columns move, [build_buysheet.py:79-85](build_buysheet.py#L79-L85)
  (`DROPDOWN_FIELDS`) needs an edit.

## Runtime setup (required to run)

1. **Python deps.** `anthropic`, `pymupdf` (fitz), `openpyxl`, `python-dotenv`,
   `Pillow`. No `requirements.txt` is included. A sibling repo's venv at
   `/Users/pranavsharma/Files/5 phase kith sheet agent /.venv` was used to
   verify the pipeline.
2. **API key.** `ANTHROPIC_API_KEY` in a project-local `.env` (loaded via
   `dotenv` from each entrypoint; `.env` is `.gitignore`d).
3. **Template.** `BUYSHEET_template.xlsx` must exist one directory above
   the repo root — [build_buysheet.py:56-57](build_buysheet.py#L56-L57)
   points `TEMPLATE_PATH = REPO_ROOT.parent / "BUYSHEET_template.xlsx"`.
   In the verification run this was satisfied by a symlink to
   `/Users/pranavsharma/Files/5 phase kith sheet agent /BUYSHEET_template.xlsx`.

## Verification (run on 2026-05-18)

Verified end-to-end on a 9-page Nike catalog
(`files/nike_HO26.pdf`, copied from the sibling repo).

```
python run_pipeline.py files/nike_HO26.pdf --workers 5
```

Result:

| Step      | Time   | Output                                            |
|-----------|--------|---------------------------------------------------|
| classify  | 20.8 s | 9/9 pages labelled `product`                      |
| extract   | 59.4 s | 110 products kept, 2 SKU false-positives rejected |
| build     | 131.1 s| 397 unique dropdown lookups, 110 rows written     |
| **total** | 211.3 s| `/Users/pranavsharma/Files/BUYSHEET_files.xlsx`   |

Spot check of written rows confirmed MG (M-Footwear / W-Footwear), SG
(Sneakers), SSG (Court / Running), Standard Color (Black / White / Brown),
and INTRO DATE (OCT / NOV) all resolved correctly. `REVIEW` sheet shows
no flagged pages. Cache directory populated at [cache/](cache/) for
follow-up runs.

## Bug fixed during verification

[`_render.py`](_render.py) only enforced a byte-size cap (3.6 MB) on the
encoded PNG before sending to Anthropic. Anthropic also rejects images
where either dimension exceeds 8000 px — the Nike catalog rendered at
9582×5475 at 300 DPI, well under the byte cap but over the dimension cap,
so every page returned `400 invalid_request_error: image dimensions exceed
max allowed size: 8000 pixels`. Added a `MAX_IMAGE_DIM = 8000` clamp
([_render.py:24-37](_render.py#L24-L37)) that proportionally fits the
longest side before the byte-size loop runs. After the fix the same
9-page run extracted 110 products with no API errors.

## Known limitations

- **Vendor detection requires a `brand_name` page.** Nike's catalog had
  no brand cover, so `default_out_path` fell back to the parent dir name
  (`files`) and the output ended up as `BUYSHEET_files.xlsx`. Override
  with `--out BUYSHEET_nike.xlsx`, or add a brand cover page upstream.
- **No `requirements.txt`.** New contributors have to discover the deps
  from imports. Adding one is a 5-minute task and not currently planned.
- **Template path is hardcoded** to one level above the repo. Moving the
  repo breaks `TEMPLATE_PATH` unless edited.
- **Stage-2 validator can only drop SKUs, never add them.** Recall is
  bounded by Stage 1 (Opus). Off-pattern or visually-hidden SKUs missed
  by Opus will be missed by the pipeline.
- **No retries beyond the SDK default** (`anthropic.Anthropic(max_retries=10)`).
  Bursty 5xx storms during build can still surface as page-level errors.
- **Cost is per-page non-deterministic.** Per CLAUDE.md, $0.10–0.25 per
  dense product page on the Opus extract; the Sonnet validator adds
  ~$0.005. The vocab_map cache amortizes dropdown costs across runs.

## File map (one-liner per module)

| File                                            | Role                                                                                    |
|-------------------------------------------------|-----------------------------------------------------------------------------------------|
| [_render.py](_render.py)                        | PDF→PNG render with byte + dimension caps for Anthropic image limits.                   |
| [classify_pages.py](classify_pages.py)          | Step 1: per-page Sonnet vision classify + context fold.                                 |
| [extract_products.py](extract_products.py)      | Step 2: per-page Opus extract (recall) + Sonnet SKU validator (precision).              |
| [vocab_map.py](vocab_map.py)                    | LLM mapper from vendor wording → closed dropdown vocab; cached per field.               |
| [build_buysheet.py](build_buysheet.py)          | Step 3: walk per-page JSON, write rows + REVIEW sheet, fan-out vocab lookups.           |
| [run_pipeline.py](run_pipeline.py)              | One-command orchestrator: classify → extract → build with `--skip` partial-rerun.       |
| [probe_products.py](probe_products.py)          | Standalone single-page Opus probe — not in the pipeline, kept for prompt iteration.     |

## Next sensible asks (not in scope yet)

- Add a `requirements.txt` (or `pyproject.toml`) so deps are explicit.
- Make `TEMPLATE_PATH` configurable via CLI flag or `BUYSHEET_TEMPLATE`
  env var; today moving the repo silently breaks `build_buysheet.py`.
- Capture the verified-runtime Python version (3.13.x from the sibling
  venv) in a pin file so 3.14+ users don't trip on missing wheels.
- A small `--smoke` mode that runs through one cached page set to confirm
  template + env without burning real LLM calls.
