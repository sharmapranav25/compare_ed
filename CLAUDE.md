# compare_ed — agentic per-page buy-sheet pipeline

A self-contained agentic pipeline that converts vendor wholesale shoe
buy-sheet documents into a filled `BUYSHEET_<vendor>.xlsx`. Single-doc
path: one PDF → three commands → one spreadsheet. Multi-doc path: N
docs (PDF + Excel) → one merged spreadsheet, with per-field priority
following CLI order.

One Opus VLM call per product page for visual extract; an Anthropic
text-layer check via PyMuPDF replaces a second LLM call when the PDF
has readable text; one Sonnet call per unique dropdown cell with
on-disk caching that makes re-runs effectively free.

## Pipeline (one CLI for everything)

```bash
# Single doc (the trivial N=1 case)
python run_pipeline.py <doc.pdf>
python run_pipeline.py <doc.pdf> --workers 8 --model opus
python run_pipeline.py <doc.pdf> --skip-analysis

# Multiple docs (priority = left-to-right argument order)
python run_pipeline.py <doc1.pdf> <doc2.xlsx> [...] [--vendor NAME]

# Step by step (debugging single-doc internals)
python classify_pages.py  <pdf>
python extract_products.py <pdf>
python build_buysheet.py   <pdf> --out BUYSHEET_<vendor>.xlsx
```

`run_pipeline.py` is a thin shim that delegates to `worker.run_multi`
— same effect as `python -m worker.run_multi <args>`, kept under the
familiar name. Each step is resumable: a crash leaves prior page JSONs
intact and reruns skip what's already done. `--force` re-runs. See
[worker.md](worker.md) for the design.

## What each file / folder does

| File / Folder | Role |
|---|---|
| `_render.py` | Shared PDF→image renderer with byte cap (3.6 MB) AND dimension cap (8000 px) for Anthropic image limits. **Adaptive encoder**: tries `ENCODE_STRATEGIES` (PNG → JPEG-q85 today) in order, downscaling 75% per pass × 4 passes within each strategy. PNG-friendly catalog pages stay PNG; photo-heavy decks (1920×1080 presentation slides) fall through to JPEG. Returns the actual saved path; caller derives the API `media_type` from the suffix via `media_type_for()`. Cached files in either format short-circuit re-render. Used by classify + extract. |
| `classify_pages.py` | **Step 1.** For every page: render → Sonnet vision → one of `category` / `brand_name` / `product` / `index_or_other` / `unknown`. On `category`, captures the section name; on `brand_name`, the vendor. Propagates a running `prev_context` (vendor, current_section) into each page's JSON. Parallel LLM calls; serial context fold. |
| `extract_products.py` | **Step 2.** For each `product`-labelled page: Stage 1 Opus visual extract (recall) → deterministic text-layer verify → Stage 2 Sonnet text-only SKU validator if and only if the text-layer check could not run (image-only PDF, PyMuPDF failure, or empty text). On the deterministic path, every VLM field is substring-matched against the PDF text layer; SKU mismatches drop the row, price mismatches null the value, description/color/intro mismatches set `verification: "unverified"`. |
| `vocab_map.py` | LLM-driven vendor wording → closed dropdown vocab. One Sonnet call per unique `(field, key)`, returning `{value, confidence}`. Three-branch caller policy:<br>• `confidence=low` → cell empty<br>• `value` matched → write canonical (orange fill)<br>• `value` null + raw present → write raw vendor value (overrides dropdown). Cache persisted at `cache/<field>.json`. |
| `build_buysheet.py` | **Step 3.** Reads every `<doc>.pages/<NN>.json`, walks in page order, runs `vocab_map.map_to_dropdown` per dropdown cell (deduped + parallel, openpyxl write serial), produces `BUYSHEET_<vendor>.xlsx`. Pages with errors / 0 products / verification issues / missing text layer get yellow STYLE# tinting + a row in the REVIEW sheet. When any product carries an `image_path`, column A is widened and each row's `OneCellAnchor`-anchored thumbnail is embedded inline. |
| [`detect/`](detect/) | Optional footwear detection. `shoe.py` lazy-loads a YOLO-family checkpoint (weights at `models/yolov8s-worldv2.pt` or `$DETECT_WEIGHTS`), filters detections to footwear class names (or uses YOLO-World's `set_classes(["shoe","sneaker","boot","sandal"])` when no per-class names are found), sorts bboxes in row-banded reading order, crops to `<doc>.pages/NN.crops/MM.png`, writes an `annotated.png` (page with numbered red boxes for the matcher) and `manifest.json` per page for resumability. Runs SERIALLY before the extract thread pool (ultralytics `.predict()` is not thread-safe) and ONLY on single-doc runs — multi-doc image merging is out of scope. Per-page YOLO `imgsz` is picked by a Sonnet vision probe that classifies the page as `low` / `medium` / `high` density (mapped to 1280 / 1920 / 2560) so dense grids and hero-shot layouts each get the right inference scale. Skips silently when `ultralytics`/weights are missing. |
| `probe_products.py` | Standalone single-page Opus probe. Not in the pipeline — kept for prompt iteration / one-off testing. |
| [`deterministic_check/`](deterministic_check/) | `verify.py` — text-layer substring verifier. Sits between extract Stage 1 and Stage 2, dropping VLM hallucinations against the PDF's embedded text stream. Per-field policy: sku miss → drop row; cost/retail miss → null value; description/color/intro miss → keep + mark `unverified`. |
| `run_pipeline.py` | User-facing CLI shim. Delegates to `worker.run_multi.main()` — kept under the familiar name so muscle memory still works. |
| [`analysis/`](analysis/) | `fill_rate.py` — pure-local report (no LLM) over a finished `.pages/`. Counts kept-vs-candidates, drop breakdown by stage, per-field fill rate, verification flag counts, plus rolled-up LLM spend by step. `usage.py` — token + cost accounting helpers used by every step. Run by the worker automatically (suppress with `--skip-analysis`). |
| [`worker/`](worker/) | **The single source of truth for the pipeline.** Handles both single-doc and multi-doc inputs uniformly (N=1 is the trivial case). Per-format adapters in `formats/`; `merge.py` joins by canonicalized SKU with first-non-empty-wins per field in priority order; `build_merged.py` reuses `build_buysheet` helpers without modifying it. Invoked via `run_pipeline.py` or directly via `python -m worker.run_multi`. Full design in [worker.md](worker.md). |

## Storage layout per doc

The pipeline creates a sibling `.pages/` directory next to each input doc:

```
<doc-stem>.pages/
  01.png    01.json    ← classify writes label + prev/context_after
  02.png    02.json    ← extract appends products, rejected_candidates,
  ...                    text_layer_present, verification dicts, usage
  _build_usage.json    ← vocab_map token accounting (written by build)
```

Excel docs (via the worker) skip the PNG, write a single synthetic
`00.json` with `page_no: 0`, `label: "product"`, and every product
field tagged `verification: "deterministic"`.

JSON shape per product page:
```json
{
  "page_no": 7,
  "label": "product",
  "prev_context":  {"vendor": "CONVERSE", "current_section": "UNISEX"},
  "context_after": {"vendor": "CONVERSE", "current_section": "UNISEX"},
  "text_layer_present": true,
  "n_candidates": 2,
  "usage": {"classify": {...}, "extract": {...}, "validate": null,
            "match": {...}},
  "products": [
    {
      "sku": "A24021 C", "description": "JACK PURCELL", "color": "...",
      "cost": "WHSLE: $64.44", "retail": "MSRP: $120.00",
      "intro_date": "Sep-15", "gender_hint": "GENDER: UNISEX",
      "image_path": "/.../A24021.pages/07.crops/02.png",
      "verification": {"sku":"ok","cost":"ok","retail":"ok",
                       "description":"ok","color":"ok","intro_date":"ok"}
    }
  ],
  "rejected_candidates": []
}
```

Sibling crop manifest `<NN>.crops/manifest.json`:
```json
{
  "bboxes": [[482, 336, 1342, 873], [2204, 410, 2687, 716], ...],
  "crop_paths": ["/.../NN.crops/01.png", ...],
  "imgsz": 1920,
  "imgsz_reasoning": "medium: 10 distinct shoes in a moderate grid"
}
```

`verification` markers: `"ok"` = substring matched the text layer;
`"unverified"` = VLM said it, text layer didn't (kept anyway);
`"not_in_text_layer"` = field was cleared (price-only policy);
`"deterministic"` = sourced from Excel via openpyxl (no VLM involved).

`image_path` is set by the optional detect+match pre-pass so a buyer can
scan thumbnails in the BUYSHEET. The flow per product page:

1. **Density picker** (Sonnet) — one vision call classifies the page as
   `low` / `medium` / `high`, picking YOLO `imgsz` of 1280 / 1920 / 2560.
2. **YOLO detection** — finds N candidate shoe bboxes at the chosen
   resolution, crops each to `<NN>.crops/MM.png`, draws numbered red
   boxes on the page → `<NN>.crops/annotated.png`.
3. **Matcher** (Sonnet) — one vision call shows the annotated page +
   the extracted SKU list, asks which numbered box belongs to which
   SKU. Hero/lifestyle/marketing shots are correctly skipped.
4. Each SKU's resolved crop path lands as `image_path` on the product.

Pages where the matcher finds no usable assignments get
`image_association: "no_match"` (surfaced in REVIEW); the per-call
spend is tracked in `usage.match` alongside extract/validate.

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
| **A Image** | crop at `image_path` | optional thumbnail (single-doc only; column appears only when at least one row has one) |

For every dropdown cell, vocab_map decides: confident match → canonical
(orange fill marks LLM-resolved); confident "nothing fits" → raw vendor
value verbatim (overrides the dropdown); low confidence → leave empty.

## Two paths through extract (deterministic-first)

Per product page, the flow is:

```
PDF page → render PNG → Stage 1 Opus visual extract (always)
                                ↓
                deterministic_check.verify_against_text_layer
                                ↓
        ┌──────────── text layer present? ────────────┐
        │ YES (digital PDF, Excel-derived PDF)        │ NO (scanned, OCR garbage, PyMuPDF raises)
        ↓                                             ↓
   per-field substring match,                    Stage 2 Sonnet text-only SKU validator
   keep / null / mark unverified                 drops obvious non-SKUs (prices, sizes, headers)
```

The verifier replaces the Stage 2 LLM call on the deterministic path,
so most pages cost one Opus call (~$0.10–0.25 / dense page); only
scanned PDFs and OCR-garbage pages fall through to Sonnet (~$0.005).
The verifier never adds SKUs — it can only filter or null what the VLM
produced.

## What this repo does NOT include

- `BUYSHEET_template.xlsx` — referenced by `build_buysheet.py` via
  `REPO_ROOT.parent / "BUYSHEET_template.xlsx"` (one directory above
  this repo). Supply your own, or edit `TEMPLATE_PATH`. The template's
  `Product Data` sheet supplies the dropdown vocabs at runtime.
- The legacy `phase1/` → `phase5/` deterministic pipeline (kept in the
  original sibling repo for back-compat; not used here).
- The cross-job scheduler layer — a future orchestrator that would
  dispatch many `worker.run_multi` runs across a worker pool for
  multi-tenant use. Today's worker is per-job only; design rules in
  [worker.md](worker.md) keep that future extension trivial to add.
- Auto-priority advisor for multi-doc input (one cheap LLM probe per
  doc → priority ranking + per-field overrides) — designed in
  [worker.md](worker.md) under "deferred", not implemented.
- PPTX adapter (LibreOffice convert → existing PDF path). Convert
  externally for now and feed the resulting PDF to the worker.

## Caching

`cache/<field>.json` files are keyed by lowercase normalized "primary
signal" (description, color, date text). Cross-doc, cross-vendor cache
hits save real money — adidas describing "STAN SMITH" populates the
same key nike's catalog might also hit. The worker's sequential-across-
docs default warms this cache between docs in one run; that's why
parallel-across-docs is opt-in. Don't delete unless you specifically
want fresh LLM calls.

## Tweak surface

- **Prompts** — every system prompt lives at the top of its file:
  `CLASSIFY_SYSTEM` in `classify_pages.py`, `EXTRACT_SYSTEM` +
  `SKU_VALIDATE_SYSTEM` in `extract_products.py`, `SYSTEM` in
  `vocab_map.py`. Edit in place, no other code knows.
- **Models** — `MODEL` constants at the top of each file. `--model opus|sonnet`
  on `extract_products.py` and `worker.run_multi` for the Stage 1 visual extract.
- **Worker count** — `--workers N` on every step (default 5).
- **Excel header synonyms** — extend `SYNONYMS` in
  [worker/formats/excel_adapter.py](worker/formats/excel_adapter.py).
- **Merge priority** — left-to-right CLI arg order to `worker.run_multi`.
  Per-field overrides (a doc winning specific fields regardless of
  position) require the deferred manifest layer.
- **Failure handling** — pages that error get an `error: "..."` field
  in their JSON. `build_buysheet.py` flags them in the xlsx (yellow
  STYLE# + REVIEW sheet). The worker's REVIEW sheet adds a `source`
  column so multi-doc flags tell you which input file owns the issue.
- **Footwear detection** — install `ultralytics` and drop a YOLO
  checkpoint at `models/yolov8s-worldv2.pt` (ultralytics auto-downloads
  this stock checkpoint on first use of `YOLO("yolov8s-worldv2.pt")`,
  or override via `$DETECT_WEIGHTS`). Auto-runs on single-doc PDFs;
  skipped silently when missing. Multi-doc runs never call detection.
- **Density picker prompt** — `YOLO_SETTINGS_SYSTEM` in
  `extract_products.py`. Adjusts how the Sonnet probe classifies pages
  as low/medium/high. The density → imgsz mapping is `_DENSITY_TO_IMGSZ`
  in the same file (default 1280 / 1920 / 2560).
- **Matcher prompt** — `IMAGE_MATCH_SYSTEM` in `extract_products.py`.
  Adjust if the matcher routinely assigns hero/lifestyle shots to SKUs
  (or vice versa).
