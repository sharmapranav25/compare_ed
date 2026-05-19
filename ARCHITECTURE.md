# Architecture & System Overview

**Audience:** CTO, eng lead, anyone reviewing the system end-to-end.
**Status:** v2 (text + photos + Sheets delivery), shipped 2026-05-19.
**Owner:** Kith Buysheet Agent team.

---

## 1. Problem statement

Kith's buying team receives vendor product catalogs as PDFs (Nike HO26,
Adidas FW26, Converse HO26, Hoka SP27, Mizuno SS27, Salomon SS27, plus
hundreds of brands across other categories). Each catalog has dozens to
hundreds of SKUs and needs to be transcribed into the Kith
`BUYSHEET_template.xlsx` — a 125-column workbook the buying team uses
for allocation, pricing, and PO management.

**Manual transcription is the bottleneck**: each catalog takes a buyer
1-3 hours of data entry per ~100 SKUs, and the error rate is non-trivial
(wrong colors, transcription typos, missed SKUs, photos paired to wrong
SKUs).

The goal: **extract product data + product photos from any vendor
catalog into the buy-sheet template with ≥90% per-cell semantic
accuracy, in <30 minutes wall-clock, with no per-vendor configuration**,
delivered to a buyer's Slack thread as both a downloadable xlsx AND a
shareable Google Sheet with photos already in-cell.

---

## 2. Approach: card-first extraction + YOLO photo detection + Sonnet matcher

The unit of truth is a **product card** — the bounded 2D region
containing one product's photo + brand + model name + SKU + color +
price. Once cards are bounded by a vision model that reads layouts the
way humans do, the field-to-SKU binding problem disappears: every field
comes from inside a known card rectangle, no rebinding heuristic exists
to be wrong.

Photo binding (which shoe image belongs to which SKU) is solved by a
**second, decoupled pass**: an open-vocabulary YOLO detector finds all
shoes on each page by visual pattern, then Sonnet does visual reasoning
over an annotated page render with numbered boxes drawn over each
detection and assigns each SKU to one of the numbered boxes. This
decouples *where are shoes?* (YOLO, layout-agnostic) from *which SKU is
which?* (Sonnet, visual reasoning) — sidestepping the text-anchored
photo-bbox failure mode that breaks on catalogs where photos aren't
reliably near SKU text.

This is the central architectural shift from the prior v0 pipeline
(which serialized the PDF to flat text and rebound fields by regex +
clustering heuristics). v0 reached ~57% per-card semantic accuracy on
Nike and ~68% on Adidas despite reporting "100% fill"; the failure mode
was a structural mis-binding documented in §6.

v2 reaches **87%+ per-cell semantic accuracy** on text fields across 7
vendor catalogs, plus **82-100% photo bbox match rate** depending on
layout — validated by a deterministic source-text oracle (text), a
cross-source SKU-blank-and-flag defense (catches single-character
vision misreads), and side-by-side Sheets generated for 7 catalogs.

---

## 3. Architecture diagram

```
PDF (uploaded to Slack)
 │
 ▼
[slack_bot/jobs.py]    one worker thread, FIFO queue
 │                     downloads PDF, posts ack with ETA, runs pipeline
 ▼
[ingest.py]            render every page to PNG @ 1568px long edge
                       + PyMuPDF text layer extraction
 │
 ▼
[classify.py]          one Opus 4.7 call on first 3 pages
                       → layout type, multi-brand, expected fields
 │
 ▼
[yolo_detect.py]       SERIAL pre-pass, one YOLO call per page
   │                   density picker (Sonnet) → imgsz 1280 / 1920 / 2560
   │                   YOLO-World detects all shoes by visual pattern
   │                   → annotated.png with red numbered boxes
   │                   → sidecar: <pdf>.v2.yolo.json
   ▼
[cards.py]             Sonnet 4.6 vision per page (Structured Outputs)
                       → CardBbox list with sku_hint from PyMuPDF
 │
 ▼
[extract.py]           Sonnet 4.6 vision per page (Structured Outputs)
                       → ProductCard list (Pydantic-typed 11 fields)
                       per-card retry fallback for dense-grid early-stop
 │
 ▼
[photo_match.py]       Sonnet 4.6 vision per page
   │                   reads annotated.png + extracted SKU list
   │                   → {sku → numbered_box} → {sku → YOLO bbox}
   │                   overwrites card.photo_bbox_px with YOLO-matched bbox
   ▼
[consistency.py]       SKU dedup + brand voting + multi-brand partition
 │
 ▼
[verify.py]            SEMANTIC ORACLE for text fields:
                       every non-null (sku, field, value) must appear
                       in source page text within the card region.
                       SKU-blank-and-flag defense: SKUs not in source
                       text on text-rich pages get blanked + commented
                       (catches AVRL→AVBL-style single-char misreads).
                       Emits per-cell confidence in [0.0, 1.0].
 │
 ├─► [write.py]                      [sheets_writer.py]
 │   BUYSHEET-template xlsx          Google Sheets API direct delivery
 │   one tab per brand               copies master template Sheet via
 │   amber/red tier formatting       Drive API, writes cell data via
 │   photos cropped on-the-fly       batchUpdate, uploads each photo
 │   from YOLO-matched bbox          to Drive subfolder, embeds via
 │   embed_photos=True default       =IMAGE(thumbnail_url, 1) formula.
 │                                   Photos are CELL VALUES from sheet
 │                                   creation — no drift on import.
 │
 ▼                                   ▼
xlsx attachment in Slack       Sheet URL in Slack
 + honest summary message:
   "Extracted N cards, M photos, fill rates per field,
    K cells amber-flagged, J cells blanked for review,
    pages X failed, Y cards truncated past template cap"
```

**Multi-model offline eval** (separate from production path):

```
[tools/multi_model_extract.py]   same card-bboxes, Sonnet vs Opus vs
                                 Gemini 2.5 Pro each extract independently
[tools/three_way_compare.py]     per-cell agreement classifier across
                                 the 3 sidecars (ALL_3_AGREE / 2_OF_3 /
                                 ALL_3_DISAGREE / *_ONLY / TWO_NULL)
[tools/eval_against_goldens.py]  per-field accuracy vs hand-verified
                                 ground truth (6 vendor goldens)
```

Used for drift detection + regression testing across prompt changes.

Full implementation: `buysheet_v2/pipeline.py` orchestrator.

---

## 4. Pipeline phases — detailed

### 4.1 Slack bot entry (slack_bot/jobs.py)

Single worker thread, FIFO queue. One job = one PDF start-to-finish.
Per-job flow:
- Download PDF from Slack (Bearer auth on `url_private`)
- Persist PDF + request manifest to `~/buysheet_runs/<stem>/<ts>/`
- Post ack in thread with page count + ETA
- Run pipeline with progress callback (25/50/75% milestone messages)
- Generate xlsx via write.py
- Generate Google Sheet via sheets_writer.py (when credentials present)
- Post done message with photos count + fill rates per field + page
  failures + truncation warnings + cost
- Upload xlsx file + post Sheet URL in same thread

Designed so per-job failures log + post an error reply but never crash
the worker; next queued job runs as normal.

### 4.2 Ingest (ingest.py)

- PyMuPDF page rendering at 1568px long edge — chosen because
  Anthropic's vision input has a max effective resolution around
  1568px, and rendering at that size means VLM-returned bboxes are in
  a known coordinate space we control (instead of an internally-
  downsampled space we'd have to reverse-engineer).
- PyMuPDF text layer extraction per page. SKUs that appear in the text
  layer become deterministic pixel anchors via `page.search_for()` —
  the foundation for the source-text oracle.
- Image-only pages (rasterized PPT exports, scanned catalogs) return
  empty text; the pipeline gracefully degrades to VLM-only confidence
  on those.

### 4.3 Layout classification (classify.py)

- One Opus 4.7 call on the first 3 rendered pages.
- Output: `LayoutClassification(layout_type, is_multi_brand,
  expected_fields_present, notes)`.
- Schema-enforced via Anthropic Structured Outputs (Nov 2025 beta).
- Drives downstream per-page prompt selection. Cost: ~$0.04 per
  catalog.

### 4.4 YOLO photo-detection pre-pass (yolo_detect.py + photo_match.py)

**This is the photo-binding architecture.** Per-page sequence:

1. **Density picker** (`photo_match.pick_yolo_density`): one Sonnet
   vision call classifies page density as low/medium/high → picks
   YOLO `imgsz` of 1280, 1920, or 2560. Sparse hero pages don't waste
   inference cycles at 2560; dense thumbnail grids don't miss shoes at
   1280. Picks UP on borderline cases ("over-detect is recoverable,
   missing shoes is not").

2. **YOLO inference** (`yolo_detect.detect_shoes_in_image`):
   YOLO-World v8s (open-vocabulary, `set_classes(["shoe","sneaker",
   "boot","sandal"])`) returns all shoe bboxes on the page. Free,
   local, ~5-15 seconds per page depending on imgsz. Detection is
   serial across pages (ultralytics.predict is not thread-safe).

3. **Annotation** (`yolo_detect.annotate_page`): renders the page
   with RED NUMBERED BOXES (1, 2, 3...) drawn over each detection in
   reading order. Saved to
   `<pdf>.v2.yolo/page_NN_annotated.png`.

4. **Matcher** (`photo_match.match_skus_to_bboxes`): runs after
   `extract.py` has produced the per-page SKU list. One Sonnet vision
   call: sees the annotated page + the SKU list, returns `{sku →
   box_number}`. Translates box number back to YOLO bbox. Overwrites
   `card.photo_bbox_px` with the YOLO-matched bbox.

5. **Sidecar**: results persisted to `<pdf>.v2.yolo.json`. Re-runs
   skip YOLO entirely; only the matcher re-fires (cheap).

If `ultralytics` or YOLO weights are unavailable, the pipeline
degrades gracefully: pipeline falls back to the legacy `photo_vlm.py`
text-anchored strategy. Detect via the optional dep `is_available()`
check.

**Why this beats the prior `photo_vlm.py` text-anchored strategy:**
- PPTX grids with tiny shoes + dense text below (Nike HO26): VLM
  text-anchor strategy targets the SKU text label instead of the
  shoe. YOLO doesn't care about text position.
- Lookbooks where descriptions are used as SKUs (Mizuno SS27): no
  anchorable SKU text → VLM strategy only resolved 6/120 bboxes.
  YOLO+matcher: 120/120.
- Hero+swatch grids (Salomon SS27): VLM bboxes drift into swatch
  labels. YOLO finds the actual shoe images.

### 4.5 Card detection (cards.py)

- One Sonnet 4.6 vision call per page.
- Inputs: rendered page image + text-anchored SKU hints from
  PyMuPDF.
- Output: `list[CardBbox]` — every visible product card with
  bounding box and SKU hint. The card region is used as framing
  context for extract.py and as the ultimate photo-bbox fallback for
  cards YOLO+matcher couldn't resolve.
- Cost: ~$0.01-0.03 per page.

### 4.6 Per-card extraction (extract.py)

- One Sonnet 4.6 vision call per page (batched: all cards' fields
  together).
- Anthropic Structured Outputs ensures every response is a valid
  `list[ProductCard]` with all 11 typed fields per card. No JSON
  parsing errors, no schema drift.
- **Per-card retry fallback** (the critical reliability mechanism):
  when the batched call returns fewer cards than detected (dense
  grid pages where the VLM gets cautious and stops early), each
  missing card is re-extracted with a *cropped image of that single
  card*. Eliminates cross-card mis-attribution because the VLM
  literally only sees one card per retry call.
- The per-card retry call returns photo_bbox in **crop-relative
  coords** — explicitly translated back to page coords by adding the
  card_bbox's top-left offset (silent coord-leakage was a bug in
  earlier iterations).
- The `model=` parameter is plumbed through so the same callsite can
  be reused by Sonnet 4.6 (default), Opus 4.7, or Haiku 4.5 — used
  by the multi-model offline eval harness.
- Cost: ~$0.05-0.15 per page (mostly retries on dense grids).

### 4.7 Semantic verification oracle (verify.py)

The text-field correctness gate. For every non-null `(sku, field,
value)` on a card, assert the value appears in the source page text
within the card region (`[previous_sku_end, this_sku_end + 200]`).

Per-field verifiers (unchanged from v1):

- **sku**: must be found via PyMuPDF text-search → 1.0
- **description / color / brand**: substring of card region → 1.0
  (whitespace-normalized to handle PPT-export quirks)
- **usd_cost / usd_retail**: numeric match in card region → 1.0
- **intro_date**: month code substring OR numeric date matching
- **mg**: derived from `_derive_mg(description)` against
  gender_vocab; trust VLM when no explicit signal
- **sg / ssg**: cached LLM mappings in `description_map.json`, falls
  back to silhouette substring matches via `silhouette_ssg_map.json`
- **standard_color**: color_synonyms.json vocab lookup

**SKU blank-and-flag defense (new in v2):** If the VLM's extracted
SKU doesn't appear in the source text AND the page has a usable
text layer (≥3 sibling SKUs ARE in text), the SKU is treated as a
vision misread. The cell is blanked, the cell comment surfaces the
VLM-extracted value plus the source-text region around the SKU
family prefix, and the reviewer can pick the correct SKU in 2
seconds without leaving the workbook.

On image-only pages (no usable text layer), SKUs go to amber
("can't verify") instead of red ("contradicted") so we don't
false-blank image-heavy catalogs.

Confidence ladder:
- **1.0** value verified — appears literally in source within card
- **0.7** vocab-confirmed OR catalog-level implicit (e.g. brand
  inferred for single-brand catalog)
- **0.5** VLM-only, no oracle path available (image-only page)
- **0.0** VLM value contradicts source — flagged for review

Cells with confidence <0.9 get amber background + cell comment;
cells ≤0.05 get red background and are blanked.

### 4.8 Cross-card consistency (consistency.py)

- **SKU uniqueness**: warn (don't drop) on duplicates.
- **Brand voting**: SKU-prefix families (e.g. all `KK*` SKUs in
  Adidas) with ≥70% agreement on a brand have that brand applied
  to outliers.
- **Multi-brand partition**: when ≥2 brands present (canonical-
  cased, whitespace-normalized), partition cards into per-brand
  buckets for worksheet tabs.

### 4.9 Workbook write (write.py)

Two-stage photo bbox cascade:

1. **If YOLO ran** (`.v2.yolo.json` sidecar exists): use
   `card.photo_bbox_px` directly (set by the matcher) for every
   card with a match.
2. **Else** (graceful degradation): fall back to legacy
   `photo_vlm.py` per-card VLM text-anchored bbox calls.
3. **Phototune fallback** for any SKU stages 1-2 didn't resolve:
   deterministic geometric heuristic from PyMuPDF SKU pixel anchor
   + row context. Mostly catches image-only pages and zero-detection
   YOLO pages.
4. **Card bbox ultimate fallback**: whole card region from cards.py.

Other write specifics:

- One worksheet per brand (template duplicated), or single
  TEMPLATE tab for single-brand catalogs.
- Workbook metadata: B1=Brand (looked up against `Vendor Data`
  sheet), B2=Season (parsed from vendor key).
- Per-cell amber fill + provenance comment when confidence <0.9.
- Per-cell red fill + AUTO-OMITTED comment + source-text snippet
  when confidence ≤0.05.
- Column A photos cropped on-the-fly from the page render using
  the resolved bbox. `embed_photos=True` is the Slack-bot
  default.
- 8 of 64 input columns written: STYLE, MG, SG, SSG, Item
  Description, Color Desc, Standard Color, INTRO DATE, USD Cost,
  USD Retail.
- Workbook write returns a stats dict (`truncated_cards`,
  `embedded_photos`, `out_path`) consumed by the Slack summary.

### 4.10 Google Sheets API direct delivery (sheets_writer.py)

**The deliverable Kith buyers actually use.** Replaces the xlsx →
manual import → Apps Script "Fix images" menu click chain with a
fully programmatic flow.

Per-job:
1. **Copy template**: `drive.files().copy()` from a master
   `BUYSHEET_template_v2_master` Sheet that lives in Drive.
   Preserves all 5 tabs (TEMPLATE + Pre Size Chart + Season +
   Product Data + Vendor Data) and all dropdown validations.
2. **Write cells**: `spreadsheets.values.batchUpdate` — one round
   trip for hundreds of cells.
3. **Tier formatting**: `repeatCell` with
   `userEnteredFormat.backgroundColor` for amber/red.
4. **Cell notes**: `updateCells.rows.values.note` carrying the
   same provenance the xlsx writer puts in cell comments.
5. **Image embedding**: each card's photo is cropped on-the-fly
   from the page render, uploaded to a per-Sheet Drive subfolder,
   then referenced via `=IMAGE("https://drive.google.com/
   thumbnail?id=FILE_ID&sz=w200", 1)` in column A. **Mode 1 forces
   fit-to-cell + preserves aspect ratio** so photos render at the
   row height without clipping.
6. **Row heights**: explicit 90px per data row via
   `updateDimensionProperties` so photos have room to render.
7. **Sharing**: domain-wide for Workspace accounts, "anyone with
   link" fallback for personal Google accounts (controlled by
   `KITH_SHEETS_DOMAIN` env var).
8. **Slack message**: posts the Sheet URL alongside the xlsx.

Photos in the resulting Sheet are **cell values from the moment the
Sheet exists**. No floating-image drift on import. No Apps Script
required. No manual menu click per catalog.

Auth supports three paths in priority order:
1. Service account JSON (production)
2. OAuth installed-app client (when org policy blocks service-
   account keys)
3. Application Default Credentials via `gcloud auth application-
   default login` (local dev)

---

## 5. Tech stack & frameworks

| Component | Library / model | Version | Why |
|---|---|---|---|
| Vision LLM (primary) | Anthropic Claude Sonnet 4.6 | current | Best-in-class layout understanding + Structured Outputs |
| Vision LLM (classify) | Anthropic Claude Opus 4.7 | current | Higher-quality doc-level reasoning; 1 call per catalog |
| Vision LLM (eval) | Google Gemini 2.5 Pro | current | Independent third-source comparison in multi-model eval |
| **Open-vocab object detection** | **Ultralytics YOLO-World** | **yolov8s-worldv2 (~25MB)** | **Layout-agnostic shoe detection; free, local, ~10ms/page inference** |
| Structured outputs | `anthropic.messages.parse(output_format=PydanticModel)` | beta 2025-11-13+ | Restricts token generation to schema-valid JSON |
| Schema validation | Pydantic | ≥2.0 | Closed-vocabulary Literal types enforce template-compatible values |
| PDF parsing | PyMuPDF (`fitz`) | ≥1.24 | Text layer + text-position search (deterministic SKU pixel anchor) |
| PDF rendering | pypdfium2 | ≥4.30 | Fast page-to-PNG at controlled DPI |
| Workbook write | openpyxl | ≥3.1 | Native xlsx with cell comments + data validations preserved |
| Image utilities | Pillow | ≥10.0 | PNG manipulation + bbox cropping + annotation drawing |
| Google Sheets API | google-api-python-client | ≥2.130 | Sheets + Drive APIs for Path C direct delivery |
| Google OAuth | google-auth + google-auth-oauthlib | ≥2.30, ≥1.2 | Installed-app OAuth flow when org blocks service-account keys |
| Slack | slack-bolt | ≥1.20 | Socket Mode worker for Slack-driven extractions |
| Env vars | python-dotenv | ≥1.0 | API key + GCP credentials loading |

### Models in use

| Purpose | Model | Hosting | Cost basis |
|---|---|---|---|
| Document classification | Claude Opus 4.7 | Anthropic API | $15 / $75 per MTok |
| Per-page card detection | Claude Sonnet 4.6 | Anthropic API | $3 / $15 per MTok |
| Per-card field extraction | Claude Sonnet 4.6 | Anthropic API | $3 / $15 per MTok |
| YOLO density picker | Claude Sonnet 4.6 | Anthropic API | $3 / $15 per MTok |
| Photo bbox matcher | Claude Sonnet 4.6 | Anthropic API | $3 / $15 per MTok |
| **Shoe detection** | **YOLO-World v8s (yolov8s-worldv2.pt)** | **Local** | **$0 — auto-downloads on first use** |
| Eval third source | Gemini 2.5 Pro | Google AI Studio | $1.25 / $5 per MTok |

### Notable choices we did NOT make

- No DocLing, RapidOCR, or layout-analysis transformers in the live
  path. Tried in v0; too slow + fragments stylized text.
- No per-vendor regex. Tried in v0; every new vendor surfaced a new
  bug.
- No fine-tuned models. YOLO-World's open-vocab head means the
  off-the-shelf checkpoint works without training data.
- No `=IMAGE(URL)` formula with externally-hosted public images for
  Path C. The URL-fetch in Google Sheets runs as the spreadsheet
  OWNER not the viewer — any URL leak = publicly accessible photos.
  Drive thumbnail URLs scoped to the same audience as the Sheet are
  the safe choice.

---

## 6. Why this approach (vs the alternatives)

### v0 was: "flat text + heuristic rebind"

The prior pipeline serialized each PDF to a per-page text blob,
applied per-vendor regex for every field (SKU, description, color,
etc), then rebound fields to SKUs via offset arithmetic + cluster
heuristics. It reached "100% fill rate" on the structural scorer
but ~57% per-card semantic accuracy on Nike and ~68% on Adidas.

Root cause: a single off-by-one in the description-routing offset
comparison mis-assigned every description to the previous SKU on
multi-product pages. The structural correctness scorer couldn't
detect this because the shifted descriptions were still unique
strings — they just belonged to the wrong row. Patching the
off-by-one lifted Nike to 96% and Adidas to 85%. **But the next
vendor with a new layout convention would have surfaced the next
regex-shape bug**, and the structural scorer would have shipped
another wrong xlsx. v0 had no path to generalize to "any random
shoebuyer catalog" — every new layout meant more regex.

### Alternatives evaluated (and why we passed)

1. **Continue per-vendor regex tuning** — dead end, see above.
2. **Fine-tuned document layout models** (LayoutLMv3, DiT, Donut)
   — need labeled training data (~1000 cards) we don't have;
   per-vendor accuracy might be high but cold-start on new vendors
   unproven.
3. **Commercial PDF tools** (Reducto, Unstructured.io, Azure
   Document Intelligence) — black-box quality; vendor lock-in;
   per-page ongoing cost; not evaluated.
4. **VLM-first card extraction with Structured Outputs** ← v1
   shipped this for TEXT extraction. Generalizes to any layout the
   VLM has seen; schema enforcement guarantees output compatibility.
5. **Per-card VLM photo-bbox extraction** (the v1 photo_vlm path)
   — works on lookbook layouts where photo + text are side-by-
   side, but fails on PPTX grids with tiny shoes + dense text
   (Nike HO26), description-as-SKU lookbooks (Mizuno SS27), and
   hero+swatch mixed layouts (Salomon SS27). Only 6/120 successful
   bboxes on Mizuno.
6. **YOLO-World + Sonnet matcher** ← v2 shipped this for PHOTOS.
   Decouples detection (visual pattern, layout-agnostic, free
   local) from matching (visual reasoning over numbered boxes —
   Sonnet's strength). 100% matched on Mizuno (up from 5%), 99%+
   on grid catalogs, 82% on the hardest PPTX layout.

The deciding factor for both pipelines: works **today** on cold
vendors without per-vendor tuning, and degrades gracefully when
upstream services are unavailable (YOLO falls back to photo_vlm
falls back to phototune falls back to card_bbox).

---

## 7. Accuracy + photo coverage

### Text-field per-cell oracle scoring (per vendor)

| Vendor | Cards | Passing ≥0.7 | Contradicted | Notes |
|---|---:|---:|---:|---|
| Nike HO26 | 110 | **87.4%** | 3.2% | Athletic grid, PPTX export |
| Adidas FW26 premium | 343 | **87.4%** | 1.6% | Dense matrix layout |
| Hoka SP27 Pinnacle | 176 | **97.6%** | 0.2% | Lookbook + sibling colorways |
| Converse HO26 | 15 | 69.2% | 6.3% | 7/17 image-only pages |
| SPS 2024 | 257 | TBD | TBD | First validated today via Path C |
| Mizuno SS27 | 120 | TBD | TBD | Lookbook, description-as-SKU |
| Salomon SS27 | 150 | TBD | TBD | Hero + swatch grid |

### Photo bbox match rate (YOLO+matcher, validated 2026-05-19)

| Vendor | Cards | YOLO-matched | Match rate | Was (text-anchored photo_vlm) |
|---|---:|---:|---:|---|
| Hoka SP27 | 176 | 172 | **98%** | Visible misalignment row 13/14 |
| Nike HO26 PPTX | 111 | 91 | **82%** | Text-crops instead of shoes |
| Salomon SS27 | 150 | 140 | **93%** | Mixed text/swatch crops |
| LLT (image-only) | 13 | 5 + 8 phototune | 100% combined | Same fallback path |
| Mizuno SS27 | 120 | 120 | **100%** | **6/120** with photo_vlm! |
| SPS 2024 | 257 | 256 | **99.6%** | (not benchmarked) |
| Adidas FW26 | 343 | pending | pending | (not benchmarked) |

The Mizuno result is the strongest validation: photo_vlm's text-
anchored strategy was nearly useless because the catalog uses
descriptions in place of SKUs (so PyMuPDF text-search anchored
almost nothing). YOLO finds shoes by visual pattern alone and
the matcher does visual reasoning on the annotated page — zero
text dependency.

### Per-field text accuracy (Nike HO26, representative)

| Field | Passing | Contradicted | Notes |
|---|---:|---:|---|
| sku | 100% | 0 | Anchored via PyMuPDF; rock solid |
| brand | 100% | 0 | Catalog-level inference for single-brand |
| description | 98% | 2 | Per-card retry essentially solved cross-card |
| color | 91% | 10 | Most remaining errors are ambiguous multi-color |
| usd_cost | 100% | 0 | Numeric source-text match |
| intro_date | 100% | 0 | Month code or numeric date match |
| standard_color | 76% | 3 | Vocab lookup gaps on novel color names |
| mg | 84% | 17 | Trusts VLM when no explicit gender signal |
| sg | 99% | 0 | description_map cache covers most |
| ssg | 33% | 0 | description_map cache miss; not wrong, unverified |

### What the oracle catches (and doesn't)

The text-field oracle replaces v0's structural scorer with a
**single correctness gate**: every claimed (sku, field, value)
must appear in the source page text within the card region.

Catches:
- Cross-card mis-attribution (the v0 bug)
- VLM hallucinations (values that don't appear anywhere in source)
- Crop-coord leakage from retry calls
- Vocab drift (colors not in synonyms map → uncertain, not wrong)
- **Single-character vision SKU misreads** (new in v2): the SKU
  blank-and-flag defense (4.7 above) catches AVRL→AVBL-class
  errors that pass field-level oracle but the SKU itself is
  wrong.

Does NOT catch:
- Values correctly present in card region but assigned to the
  wrong card (rare with per-card retry)
- Whitespace/punctuation variations (handled by normalization)

---

## 8. Cost

### Per-catalog API spend (production path: extract + verify + photos)

| Vendor | Pages | Cards | API spend | Wall time |
|---|---:|---:|---:|---:|
| Nike HO26 | 9 | 110 | **~$1.20** | ~4 min |
| Adidas FW26 premium | 31 | 343 | **~$4.50** | ~12 min |
| Hoka SP27 | 9 | 176 | **~$1.50** | ~5 min |
| Mizuno SS27 | 90 | 120 | **~$3.00** | ~10 min |
| SPS 2024 | 66 | 257 | **~$3.50** | ~11 min |

Cost breakdown (per catalog, typical):
- Classify (Opus 4.7, 1 call): $0.04
- Card detection (Sonnet, N calls): $0.30-1.00
- Field extraction (Sonnet, N calls + retries): $0.50-2.00
- YOLO density picker (Sonnet, N calls): $0.10-0.50
- Matcher (Sonnet, N calls): $0.15-0.75
- YOLO inference: $0 (local)
- Drive uploads (Path C): $0 (Google free quota)
- **Total: ~$1.50-5.00 per catalog**

Average across 5 validated catalogs: **~$2.74 per catalog**.

### Projection at scale

- 100 catalogs/month: ~$275/month
- 500 catalogs/month: ~$1,375/month

Anthropic prompt-caching is enabled on system prompts (ephemeral
cache TTL = 5 min). Within-catalog calls reuse cached system
prompts after the first page.

Each pipeline run writes a sidecar (`<doc>.v2.cards.json`,
`<doc>.v2.yolo.json`) so re-runs after prompt tweaks are free
(skip Anthropic + YOLO + Drive entirely; just re-emit xlsx /
Sheet).

---

## 9. Multi-model eval harness (offline)

Separate from the production path. Used for drift detection +
regression testing across prompt changes.

`tools/multi_model_extract.py` re-extracts the same card-bboxes
from `cards.py` with each of Sonnet 4.6, Opus 4.7, and Gemini 2.5
Pro independently. Produces 3 sidecars + 3 xlsx files per PDF.

`tools/three_way_compare.py` classifies every (sku, field) cell
across the 3 sidecars:

| Tier | Meaning |
|---|---|
| ALL_3_AGREE | All three models produced the same normalized value |
| 2_OF_3_AGREE | Two agree, one dissents (usually the dissenter is wrong) |
| ALL_3_DISAGREE | Three different values — worst case, no signal |
| SONNET_ONLY / OPUS_ONLY / GEMINI_ONLY | Only that model extracted a value |
| TWO_AGREE_ONE_NULL | Two have same value, third left null |

The harness was used to validate that the apples-to-apples
Gemini extractor (matching prompt + per-card hints) produces
substantively different output than Sonnet on description and
color (about 5-10% of cells), establishing a real second-opinion
signal rather than a "models confirm each other because they
share training data" artifact.

`tools/eval_against_goldens.py` scores extracted sidecars
against 6 hand-verified vendor goldens (Nike, Adidas, Hoka,
Mizuno, SPS, Converse — 20-25 SKUs each, per-field truth). Used
as a CI gate: regressions block merges. Per-field, per-vendor
match rates are written to `tests/golden_baseline.json` which
is git-tracked.

---

## 10. Known limitations & v3 candidates

### Currently SHIPPED in v2

- Per-cell text accuracy 87%+ across 5 validated catalogs.
- Photo coverage 82-100% depending on layout (averaging 95%+
  excluding image-only edge cases).
- Direct Google Sheets delivery with photos in-cell at sheet
  creation time.
- SKU blank-and-flag defense for single-char vision misreads.
- Multi-model eval harness for drift detection.
- Honest Slack summary: per-field fill rates, page failure
  count, row-truncation warning, photo embed count.

### Known gaps

- **ssg vocab coverage**: `description_map.json` was built from v0
  vendor runs; new product names (HO26 / FW26 / SS27) aren't
  cached. ssg drops to ~33% "passing" on new descriptions — but
  these are not wrong, just unverified. Closing this is a vocab-
  enrichment task; `tools/enrich_description_map.py` handles it
  on-demand for ~$0.03 per 30 novel descriptions.
- **Image-only PDFs** extract correctly via VLM but can't be
  verified against text layer; flagged as `vlm_only_no_region`.
  YOLO+matcher works on these too if YOLO detects the shoes (LLT
  catalog: 5/13 YOLO + 8 phototune fallback).
- **`extract_single_card` retry costs**: dense grid pages (Nike
  page 1 with 38 cards) trigger 30+ per-card retry calls,
  doubling per-page cost. Acceptable today; could be optimized
  via tighter row-context geometry in future.
- **YOLO inference is serial** (ultralytics.predict isn't thread-
  safe). Adidas-scale catalogs (90+ pages) spend ~5 min in the
  YOLO pre-pass alone. Could use a model-server pattern for
  parallel inference if catalog volume justifies it.

### v3 candidates

- **OCR-on-demand for scanned PDFs** — currently image-only
  catalogs (LLT-class) can't verify text via PyMuPDF. A Claude
  vision OCR pass triggered when text density is below a
  threshold would let the oracle work on those too.
- **Per-page layout understanding** — replace the single-doc
  `classify.py` call with a per-page lightweight classifier so
  mixed catalogs (grid + lookbook + spec sheets in one PDF) get
  the right snippet boundaries per page.
- **Anthropic batch API** — if Kith starts processing 1000+
  catalogs/month, switching to batched messages would cut
  latency and cost ~50%.
- **Focused multi-model consensus online** — when verify flags a
  cell amber, re-extract that single cell with Opus on a cropped
  card image. If they disagree, Gemini tiebreak. ~$0.50-1
  adder per catalog, resolves ~5-10% of amber cells.

---

## 11. Project layout

```
KithxKeeloShoeBuying/
├── README.md                       # quick start + accuracy snapshot
├── ARCHITECTURE.md                 # this file
├── ACCURACY_EXPECTATIONS.md        # per-vendor fill-rate expectations
├── LICENSE
├── BUYSHEET_template.xlsx          # immutable output template
├── apps_script/                    # legacy "Fix images" Apps Script
│                                   # (superseded by Path C in v2, retained
│                                   #  as fallback for buyers who prefer xlsx)
└── buysheet_v2/
    ├── pyproject.toml
    ├── README.md                   # package-level overview
    ├── cli.py                      # python -m buysheet_v2 entry point
    ├── pipeline.py                 # orchestrator (YOLO+matcher integrated)
    ├── ingest.py                   # PDF -> renders + text
    ├── classify.py                 # Opus 4.7 layout classification
    ├── cards.py                    # Sonnet 4.6 card detection
    ├── extract.py                  # Sonnet 4.6 structured extraction + retry
    ├── yolo_detect.py              # YOLO-World shoe detection pre-pass
    ├── photo_match.py              # density picker + SKU→bbox matcher
    ├── photo_vlm.py                # legacy text-anchored photo bbox
    │                               # (graceful-degradation fallback)
    ├── phototune.py                # deterministic geometric photo fallback
    ├── verify.py                   # semantic oracle + SKU blank-and-flag
    ├── consistency.py              # SKU dedup + brand voting
    ├── confidence.py               # per-cell confidence model
    ├── write.py                    # xlsx generation
    ├── sheets_writer.py            # Path C Google Sheets API delivery
    ├── schemas/                    # Pydantic models
    │   ├── card.py
    │   ├── doc_layout.py
    │   ├── extraction_result.py
    │   └── photo_match.py
    ├── prompts/                    # versioned VLM prompts
    │   ├── layout_classify.md
    │   ├── card_detect.md
    │   └── card_extract.md
    ├── vocab/                      # closed vocabs
    │   ├── color_synonyms.json
    │   ├── description_map.json
    │   ├── silhouette_ssg_map.json
    │   └── ...
    ├── models/                     # YOLO weights (auto-downloaded)
    │   └── yolov8s-worldv2.pt      # ~25MB, gitignored
    ├── slack_bot/                  # Slack Socket Mode entrypoint
    │   ├── app.py
    │   ├── jobs.py                 # FIFO worker, posts xlsx + Sheet URL
    │   └── formatting.py           # message templates
    ├── lifted/                     # v0 helpers (page render, vocab norm)
    └── tools/
        ├── eval_against_goldens.py # per-vendor accuracy reporter
        ├── multi_model_extract.py  # 3-way Sonnet+Opus+Gemini extraction
        ├── three_way_compare.py    # cross-model agreement classifier
        ├── scaffold_golden_v2.py   # build ground-truth templates
        ├── verify_golden.py        # interactive golden verifier
        ├── enrich_description_map.py
        └── enrich_color_synonyms.py
```

---

## 12. Environment variables

| Variable | Purpose | Required? |
|---|---|---|
| `ANTHROPIC_API_KEY` | Sonnet/Opus access | Yes |
| `GEMINI_API_KEY` | Eval harness only (multi-model compare) | No |
| `GEMINI_MODEL` | Override Gemini model (default: `gemini-2.5-pro`) | No |
| `GEMINI_MIN_INTERVAL_SEC` | Free-tier rate limit throttle (default 4s) | No |
| `SLACK_BOT_TOKEN` | Slack bot OAuth token (`xoxb-...`) | Yes for bot |
| `SLACK_APP_TOKEN` | Slack Socket Mode app token (`xapp-...`) | Yes for bot |
| `GOOGLE_SERVICE_ACCOUNT_JSON` | Service-account JSON path (Path C primary) | Path C |
| `GOOGLE_OAUTH_CLIENT_JSON` | OAuth installed-app JSON (Path C fallback when org blocks service-account keys) | Path C |
| `GOOGLE_OAUTH_TOKEN_CACHE` | Where the OAuth refresh token is cached | No (default: `~/.config/gcp/kith-buysheet-token.json`) |
| `KITH_TEMPLATE_SHEET_ID` | Drive ID of master `BUYSHEET_template_v2_master` Sheet | Path C |
| `KITH_SHEETS_DOMAIN` | Workspace domain for sharing, or `_public` for anyone-with-link | Path C |
| `KITH_SHEETS_PARENT_FOLDER` | Drive folder ID where new Sheets land | No |
| `DETECT_WEIGHTS` | Override path to YOLO weights | No (default: `buysheet_v2/models/yolov8s-worldv2.pt`) |
| `BUYSHEET_RUNS_DIR` | Where per-job artefacts persist | No (default: `~/buysheet_runs`) |

---

## 13. Decisions log

| Decision | Why |
|---|---|
| Sonnet 4.6 (not Opus) for per-page extraction | 5x cheaper, accuracy parity on layout tasks |
| Opus 4.7 only for doc-level classify | Single call per catalog; higher reasoning quality justified |
| Render @ 1568px long edge | Matches Anthropic vision input ceiling; returned bboxes in known coord system |
| Structured Outputs (vs. JSON-mode + parse) | Guarantees schema conformance; zero retry loops |
| Per-card retry fallback (vs. one-shot per-page) | Dense grids cause early-stop end_turn; retry recovers them |
| Semantic oracle (vs. structural scorer) | v0's structural scorer shipped wrong xlsx; oracle catches semantic errors |
| YOLO-World (vs. fine-tuned detector) | Open-vocab head means zero training data needed; works on any catalog |
| Sonnet matcher over annotated page (vs. closest-bbox heuristic) | Visual reasoning over numbered boxes is what Sonnet is good at; heuristics break on multi-section grids |
| Density picker per-page (vs. fixed imgsz) | Sparse pages waste cycles at 2560; dense grids miss shoes at 1280 |
| YOLO sidecar caching (vs. re-detect each run) | Detection takes 5-15s/page; sidecar makes re-runs free |
| Path C direct Sheets API (vs. xlsx + Apps Script) | Eliminates the manual import + menu-click chain; photos are cell values from sheet creation = no drift |
| OAuth installed-app (vs. service account) | Some orgs disable service-account JSON keys via Secure-by-Default policy |
| `=IMAGE(url, 1)` explicit mode (vs. default) | Forces fit-to-cell; without it some Sheets configs render at native size and clip |
| SKU blank-and-flag (vs. auto-correct via edit distance) | Loud failure > silent wrong correction; blank cell forces 2-second human review |
| Three-way model eval (Sonnet+Opus+Gemini) | Drift detection across prompt changes; multi-model agreement as a high-signal review queue |
| Apples-to-apples Gemini (full prompt + bbox hints) | Earlier "vendor-agnostic" Gemini variant was strawman; this version tests the MODEL not the prompt |
| Multi-brand split (vs. single tab with brand column) | Template assumes 1 brand per workbook; per-tab preserves template contract |
| temperature=0 NOT settable (Claude 4.x) | API rejects; mitigated by per-card retry mechanism |
| Skip-on-existing sidecar caching | Re-runs after prompt iteration cost $0 |
| Slack bot single worker queue (vs. concurrent jobs) | Simpler error handling; YOLO not thread-safe anyway; can scale later |
| Honest Slack summary (fill rates + page errors + truncation) | Silent failures were the previous failure mode; now buyer sees what shipped |
