# compare_ed — agentic per-page buy-sheet pipeline

A self-contained agentic pipeline that converts a multi-vendor shoe buy-sheet
PDF into a filled `BUYSHEET_<vendor>.xlsx`. One PDF → three commands → one
spreadsheet. Two VLM calls per product page (extract + validate); one
text-only LLM call per unique dropdown cell, with on-disk caching that makes
re-runs effectively free.

## Pipeline (3 commands or 1)

```bash
# One-shot
python run_pipeline.py <pdf>
python run_pipeline.py <pdf> --workers 8 --model opus
python run_pipeline.py <pdf> --skip classify extract     # rebuild xlsx only

# Or step by step
python classify_pages.py  <pdf>
python extract_products.py <pdf>
python build_buysheet.py   <pdf> --out BUYSHEET_<vendor>.xlsx
```

Each step is resumable: a crash leaves prior page JSONs intact and reruns
skip what's already done. `--force` re-runs.

## What each file does

| File | Role |
|---|---|
| `_render.py` | Shared PDF→PNG render with the 3.6 MB Anthropic image-size guard (75% downscale per pass, up to 4 passes). Used by classify + extract. |
| `classify_pages.py` | **Step 1.** For every page: render → Sonnet vision → one of `category` / `brand_name` / `product` / `index_or_other` / `unknown`. On `category`, captures the section name; on `brand_name`, the vendor. Propagates a running `prev_context` (vendor, current_section) into each page's JSON. Parallel LLM calls; serial context fold. |
| `extract_products.py` | **Step 2.** For each `product`-labelled page, a two-stage pass:<br>**Stage 1 — Opus visual extract (recall):** finds every distinct product on the page, returning sku + description + color + cost + retail + intro_date + gender_hint.<br>**Stage 2 — Sonnet text-only validator (precision):** given just the SKU strings (no image), drops obvious false positives (prices, sizes, section headers mis-classified as SKUs). Per-page JSON gets `products` (kept) + `rejected_candidates` (debug). |
| `vocab_map.py` | LLM-driven vendor wording → closed dropdown vocab. One Sonnet call per unique `(field, key)`, returning `{value, confidence}`. Three-branch policy in the caller:<br>• `confidence=low` → cell empty<br>• `value` matched → write canonical (orange fill)<br>• `value` null + raw present → write raw vendor value (overrides dropdown). Cache persisted at `cache/<field>.json`. |
| `build_buysheet.py` | **Step 3.** Reads every `<doc>.pages/<NN>.json`, walks in page order, runs `vocab_map.map_to_dropdown` per dropdown cell (deduped + parallel, openpyxl write serial), produces `BUYSHEET_<vendor>.xlsx`. Pages with errors / 0 products get yellow STYLE# tinting + a row in the REVIEW sheet. |
| `run_pipeline.py` | One-command orchestrator. Imports each step's function, sequential. `--skip {classify,extract,build}` for partial re-runs. |
| `probe_products.py` | Standalone single-page Opus probe. Not in the pipeline — kept for prompt iteration / one-off testing. |

## Storage layout per PDF

The pipeline creates a sibling `.pages/` directory next to the input PDF:

```
<pdf-stem>.pages/
  01.png    01.json    ← classify writes label + prev/context_after
  02.png    02.json    ← extract appends `products` + `rejected_candidates`
  ...
```

JSON shape per page:
```json
{
  "page_no": 7,
  "label": "product",
  "prev_context": {"vendor": "CONVERSE", "current_section": "UNISEX"},
  "context_after": {"vendor": "CONVERSE", "current_section": "UNISEX"},
  "products": [
    {"sku": "A24021 C", "description": "JACK PURCELL", "color": "...",
     "cost": null, "retail": "...", "intro_date": null, "gender_hint": "..."}
  ],
  "rejected_candidates": []
}
```

## Cell-fill policy

| Buy-sheet col | Source | Rule |
|---|---|---|
| **B STYLE #** | extracted `sku` | raw vendor value |
| **C MG** | dropdown | LLM-mapped from `description + section + gender_hint` |
| **D SG** | dropdown | LLM-mapped from `description` (silhouette/model name) |
| **E SSG** | dropdown | LLM-mapped from `description` |
| **F Item Description** | extracted `description` | raw |
| **G Color Desc** | extracted `color` | raw |
| **H Standard Color** | dropdown | LLM-mapped from `color` |
| **P INTRO DATE** | dropdown (JAN..DEC) | LLM-mapped from extracted date |
| **V USD Cost** | extracted `cost` | parsed to number; non-numeric → raw |
| **W USD Retail** | extracted `retail` | parsed to number; non-numeric → raw |

For every dropdown cell, vocab_map decides: confident match → canonical (orange fill marks LLM-resolved); confident "nothing fits" → raw vendor value verbatim (overrides the dropdown); low confidence → leave empty.

## Why two VLM calls per product page

A single extract call has to do visual scanning AND structuring at once,
which leaks recall on dense / off-pattern layouts. Splitting recall
(Stage 1 Opus visual extract) from precision (Stage 2 Sonnet text-only
validator) lets each stage focus. The validator never adds SKUs the VLM
missed — it can only drop false positives the VLM included.

Cost per page (Opus 4.7 extract + Sonnet 4.6 validator):
~$0.10–0.25 / dense product page, ~$0.005 for the validator.

## What this repo does NOT include

This was extracted from a larger 5-phase pipeline. The standalone copy
omits:

- `BUYSHEET_template.xlsx` — currently referenced by `build_buysheet.py`
  via `REPO_ROOT / "BUYSHEET_template.xlsx"`. You need to drop a copy in
  the project root one level up, or edit `TEMPLATE_PATH` to point at your
  copy. The template's `Product Data` sheet is read at runtime — that's
  where the dropdown vocabs come from.
- The legacy `phase1/` → `phase5/` deterministic pipeline (kept in the
  original repo for back-compat; not used here).

## Caching

`cache/<field>.json` files are keyed by lowercase normalized "primary
signal" (description, color, date text). Cross-vendor cache hits save
real money — adidas describing "STAN SMITH" populates the same key
nike's catalog might also hit. Don't delete unless you specifically want
fresh LLM calls.

## Tweak surface

- **Prompts** — every system prompt lives at the top of its file:
  `CLASSIFY_SYSTEM` in `classify_pages.py`, `EXTRACT_SYSTEM` +
  `SKU_VALIDATE_SYSTEM` in `extract_products.py`, `SYSTEM` in
  `vocab_map.py`. Edit in place, no other code knows.
- **Models** — `MODEL` constants at the top of each file. `--model opus|sonnet`
  on `extract_products.py` for the Stage 1 visual extract.
- **Worker count** — `--workers N` on every step (default 5).
- **Failure handling** — pages that error get an `error: "..."` field in
  their JSON. `build_buysheet.py` flags them in the xlsx (yellow STYLE#
  + REVIEW sheet).
