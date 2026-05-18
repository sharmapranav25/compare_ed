# Architecture & System Overview

**Audience:** CTO, eng lead, anyone reviewing the system end-to-end.
**Status:** v1 (text-only), shipped 2026-05-18.
**Owner:** Kith Buysheet Agent team.

---

## 1. Problem statement

Kith's buying team receives vendor product catalogs as PDFs (Nike HO26,
Adidas FW26, Converse HO26, Hoka SP27, Mizuno SS27, plus hundreds of
brands across other categories). Each catalog has dozens to hundreds of
SKUs and needs to be transcribed into the Kith
`BUYSHEET_template.xlsx` — a 125-column workbook the buying team uses for
allocation, pricing, and PO management.

**Manual transcription is the bottleneck**: each catalog takes a buyer
1-3 hours of data entry per ~100 SKUs, and the error rate is non-trivial
(wrong colors, transcription typos, missed SKUs).

The goal: **extract product data from any vendor catalog into the buy-sheet
template with ≥90% per-cell semantic accuracy, in <30 minutes wall-clock,
with no per-vendor configuration**.

---

## 2. Approach: VLM-first card extraction

The unit of truth is a **product card** — the bounded 2D region containing
one product's photo + brand + model name + SKU + color + price. Once cards
are bounded by a vision model that reads layouts the way humans do, the
field-to-SKU binding problem disappears: every field comes from inside a
known card rectangle, no rebinding heuristic exists to be wrong.

This is the central architectural shift from the prior v0 pipeline (which
serialized the PDF to flat text and rebound fields by regex + clustering
heuristics). v0 reached ~57% per-card semantic accuracy on Nike and ~68%
on Adidas despite reporting "100% fill"; the failure mode was a structural
mis-binding (described in §6 below).

v1 reaches **87% per-cell semantic accuracy** on Nike and Adidas via the
card-first architecture documented here, validated by a deterministic
source-text oracle (rather than a structural scorer that can't see semantic
errors).

---

## 3. Architecture diagram

```
PDF
 ↓
[ingest.py]      render every page to PNG @ 1568px long edge + PyMuPDF text layer
 ↓
[classify.py]    one Opus 4.7 call on first 3 pages
                 → layout type {grid, vertical_card, spec_panel, lookbook, mixed}
                 → is_multi_brand?
                 → expected fields present (informs downstream prompts)
 ↓
[cards.py]       Sonnet 4.6 vision per page (Structured Outputs)
                 → list of CardBbox {page, bbox_px, sku_hint}
                 uses text-anchored SKUs from PyMuPDF as binding hints
 ↓
[extract.py]     Sonnet 4.6 vision per page (Structured Outputs)
                 → list of ProductCard (Pydantic) with all 11 fields
                 per-card RETRY fallback: when a dense-page batched call
                 returns fewer cards than detected, re-extract missing ones
                 individually with the card image cropped (eliminates
                 cross-card mis-attribution by construction)
 ↓
[verify.py]      deterministic SEMANTIC ORACLE — for every non-null
                 (sku, field, value):
                  - locate SKU pixel offset in source page text
                  - establish card text region (bounded by previous + next SKU)
                  - assert the value appears in that region
                  - vocab lookups (color_synonyms, description_map,
                    silhouette_ssg_map) confirm mg/sg/ssg/standard_color
                 emits per-cell confidence in [0.0, 1.0]
 ↓
[consistency.py] cross-card validation
                  - SKU uniqueness
                  - brand voting per SKU-prefix family (repairs single-card mis-attribution)
                  - multi-brand workbook partition (one tab per brand)
 ↓
[write.py]       BUYSHEET-template xlsx
                  - one tab per brand (canonical-cased)
                  - 8 text columns + workbook metadata (B1=Brand, B2=Season)
                  - amber background + cell comment for confidence <0.9
                  - text-only in v1 (column A reserved for v2 image binding)
```

Full implementation: `buysheet_v2/pipeline.py` orchestrator.

---

## 4. Pipeline phases — detailed

### 4.1 Ingest (ingest.py)

- PyMuPDF page rendering at 1568px long edge — chosen because Anthropic's
  vision input has a max effective resolution around 1568px, and rendering
  at that size means VLM-returned bboxes are in a known coordinate space
  we control (instead of an internally-downsampled space we'd have to
  reverse-engineer).
- PyMuPDF text layer extraction per page. SKUs that appear in the text
  layer become deterministic pixel anchors via `page.search_for()` —
  the foundation for the source-text oracle.
- Image-only pages (rasterized PPT exports) return empty text; the
  pipeline gracefully degrades to VLM-only confidence on those.

### 4.2 Layout classification (classify.py)

- One Opus 4.7 call on the first 3 rendered pages.
- Output: `LayoutClassification(layout_type, is_multi_brand, expected_fields_present, notes)`.
- Schema-enforced via Anthropic Structured Outputs (Nov 2025 beta).
- Drives downstream per-page prompt selection. Cost: ~$0.04 per catalog.

### 4.3 Card detection (cards.py)

- One Sonnet 4.6 vision call per page.
- Inputs: rendered page image + text-anchored SKU hints from PyMuPDF.
- Output: `list[CardBbox]` — every visible product card with bounding box
  and SKU hint.
- Cost: ~$0.01-0.03 per page.

### 4.4 Per-card extraction (extract.py)

- One Sonnet 4.6 vision call per page (batched: all cards' fields together).
- Anthropic Structured Outputs ensures every response is a valid
  `list[ProductCard]` with all 11 typed fields per card. No JSON parsing
  errors, no schema drift.
- **Per-card retry fallback** (the critical reliability mechanism): when
  the batched call returns fewer cards than detected (dense grid pages
  where the VLM gets cautious and stops early), each missing card is
  re-extracted with a *cropped image of that single card*. Eliminates
  cross-card mis-attribution because the VLM literally only sees one
  card per retry call.
- The per-card retry call returns photo_bbox in **crop-relative coords**
  — these are explicitly translated back to page coords by adding the
  card_bbox's top-left offset (silent coord-leakage was a bug in earlier
  iterations).
- Cost: ~$0.05-0.15 per page (mostly retries on dense grids).

### 4.5 Semantic verification oracle (verify.py)

The single correctness gate. For every non-null (sku, field, value) on a
card, assert the value appears in the source page text within the card
region. The card region is bounded by `[previous_sku_end, this_sku_end + 200]`
in the source text.

Per-field verifiers:
- **sku**: must be found via PyMuPDF text-search → 1.0
- **description / color / brand**: substring of card region → 1.0
  (whitespace-normalized to handle PPT-export quirks)
- **usd_cost / usd_retail**: numeric match in card region → 1.0
- **intro_date**: month code substring OR numeric date matching that month
- **mg**: derived from `_derive_mg(description)` against gender_vocab;
  trust VLM when no explicit signal (avoids over-strict false positives)
- **sg / ssg**: cached LLM mappings in `description_map.json`, falls back
  to silhouette substring matches via `silhouette_ssg_map.json`
- **standard_color**: color_synonyms.json vocab lookup; high confidence
  when raw color text confirms

Confidence ladder:
- **1.0** value verified — appears literally in source within card region
- **0.7** vocab-confirmed OR catalog-level implicit (e.g. brand inferred)
- **0.5** VLM-only, no oracle path available
- **0.0** VLM value contradicts source — flagged for review

Cells with confidence <0.9 get amber background + cell comment in the
output xlsx.

### 4.6 Cross-card consistency (consistency.py)

- **SKU uniqueness**: warn (don't drop) on duplicates.
- **Brand voting**: SKU-prefix families (e.g. all `KK*` SKUs in Adidas)
  with ≥70% agreement on a brand have that brand applied to outliers.
- **Multi-brand partition**: when ≥2 brands present (canonical-cased,
  whitespace-normalized), partition cards into per-brand buckets for
  worksheet tabs.

### 4.7 Workbook write (write.py)

- One worksheet per brand (template duplicated), or single TEMPLATE tab
  for single-brand catalogs.
- Workbook metadata: B1=Brand (looked up against `Vendor Data` sheet),
  B2=Season (parsed from vendor key, e.g. `nike_ho26` → `2026-Q4`).
- Per-cell amber fill + provenance comment when confidence <0.9.
- Column A (PHOTO) reserved but empty in v1.
- 8 of 64 input columns written: STYLE, MG, SG, SSG, Item Description,
  Color Desc, Standard Color, INTRO DATE, USD Cost, USD Retail (the rest
  are buyer-input or template-calculated).

---

## 5. Tech stack & frameworks

| Component | Library | Version | Why |
|---|---|---|---|
| Vision LLM | Anthropic Claude Sonnet 4.6 | (current) | Best-in-class layout understanding; Structured Outputs guarantees schema conformance |
| Classification | Anthropic Claude Opus 4.7 | (current) | Higher-quality doc-level reasoning; only 1 call per catalog so cost negligible |
| Structured outputs | `anthropic.messages.parse(output_format=PydanticModel)` | beta 2025-11-13+ | Restricts token generation to schema-valid JSON; no parse errors, no retry loops |
| Schema validation | Pydantic | ≥2.0 | Closed-vocabulary Literal types enforce template-compatible values at extraction time |
| PDF parsing | PyMuPDF (`fitz`) | ≥1.24 | Text layer extraction + text-position search (deterministic SKU pixel anchor) |
| PDF rendering | pypdfium2 | ≥4.30 | Fast page-to-PNG render at controlled DPI |
| Workbook write | openpyxl | ≥3.1 | Native xlsx with cell comments + data validations preserved from template |
| Image utilities | Pillow | ≥10.0 | PNG manipulation (retained for v2 image-binding work) |
| Env vars | python-dotenv | ≥1.0 | API key loading |

**Notable choices we did NOT make:**
- No DocLing, RapidOCR, or layout-analysis transformers. Tried in v0; too
  slow + fragments stylized text.
- No per-vendor regex. Tried in v0; every new vendor surfaced a new bug.
- No fine-tuned models. Anthropic's hosted VLM + Structured Outputs is
  sufficient for current accuracy targets.

---

## 6. Why this approach (vs the alternatives)

### v0 was: "flat text + heuristic rebind"

The prior pipeline serialized each PDF to a per-page text blob, applied
per-vendor regex for every field (SKU, description, color, etc), then
rebound fields to SKUs via offset arithmetic + cluster heuristics. It
reached "100% fill rate" on the structural scorer but ~57% per-card
semantic accuracy on Nike and ~68% on Adidas.

Root cause: a single off-by-one in the description-routing offset
comparison (`if e <= offset and (offset - e) < 300`) mis-assigned every
description to the previous SKU on multi-product pages. The structural
correctness scorer couldn't detect this because the shifted descriptions
were still unique strings — they just belonged to the wrong row.

Patching the off-by-one lifted Nike to 96% and Adidas to 85%. **But the
next vendor with a new layout convention would have surfaced the next
regex-shape bug, and the structural scorer would have said `flag_rate=1.5%`
and shipped another wrong xlsx.** The v0 architecture had no path to
generalize to "any random shoebuyer catalog" — every new layout meant
more regex.

### Alternatives evaluated

1. **Continue per-vendor regex tuning** — dead end, see above.
2. **Fine-tuned document layout models** (LayoutLMv3, DiT, Donut) —
   need labeled training data (~1000 cards) we don't have; per-vendor
   accuracy might be high but cold-start on new vendors unproven.
3. **Commercial PDF tools** (Reducto, Unstructured.io, Azure Document
   Intelligence) — black-box quality; vendor lock-in; per-page ongoing
   cost; not evaluated.
4. **VLM-first card extraction with Structured Outputs** ← this is what
   we shipped. Generalizes to any layout the VLM has seen (which is
   "everything"); schema enforcement guarantees output compatibility;
   semantic oracle catches hallucinations.

The deciding factor: VLM-first works **today** on cold vendors without
per-vendor tuning. Fine-tuned approaches might be slightly more accurate
on familiar layouts but require investment to onboard each new vendor.

---

## 7. Accuracy breakdown

Per-cell oracle scoring across 3 vendors (110 + 346 + 15 = 471 SKUs, 4,440 cells total):

| Vendor | Cards | Passing ≥0.7 | Contradicted | Cost |
|---|---:|---:|---:|---:|
| Nike HO26 | 110 | **87.4%** | 3.2% | $0.53 |
| Adidas FW26 premium | 346 | **87.4%** | 1.6% | $1.73 |
| Converse HO26 | 15 | 69.2% | 6.3% | $0.34 |

**Per-field accuracy on Nike HO26 (representative):**

| Field | Passing | Contradicted | Notes |
|---|---:|---:|---|
| sku | 100% | 0 | Anchored via PyMuPDF; rock solid |
| brand | 100% | 0 | Catalog-level inference for single-brand catalogs |
| description | 98% | 2 | Per-card retry essentially solved cross-card mis-attribution |
| color | 91% | 10 | Most remaining errors are ambiguous multi-color cards |
| usd_cost | 100% | 0 | Numeric source-text match |
| intro_date | 100% | 0 | Month-code or numeric-date match against source |
| standard_color | 76% | 3 | Vocab lookup gaps on novel color names |
| mg | 84% | 17 | Soft verification — trusts VLM when no explicit gender signal |
| sg | 99% | 0 | description_map cache covers most |
| ssg | 33% | 0 | description_map cache miss on novel HO26 product names; not "wrong", just unverified |

Converse runs lower (69%) because 7 of 17 pages are image-only PPT
exports (no text layer); the semantic oracle can't verify values against
absent source text. Field extraction itself is comparable to the
text-layer pages — confidence just drops to `vlm_only_no_region` since
verification is impossible.

### What the oracle catches

The semantic oracle replaces v0's structural scorer with a **single
correctness gate**: every claimed (sku, field, value) must appear in
the source page text within the card region. It catches:

- Cross-card mis-attribution (the v0 bug)
- VLM hallucinations (values that don't appear anywhere in the source)
- Crop-coord leakage from retry calls
- Vocab drift (color names not in synonyms map → marked uncertain rather
  than silently wrong)

It does NOT catch:
- Values correctly present in the card region but assigned to the wrong
  card by the VLM (the contradicted% catches this when it overlaps with
  another SKU; otherwise marked passing)
- Whitespace/punctuation variations (handled by aggressive normalization)

### Cold-vendor evaluation (in progress)

`buysheet_v2/tests/holdout/` contains scaffolded test sets for Hoka SP27
and Mizuno SS27 — two vendors NOT used during pipeline development.
Human verification of these holdouts is the ship gate for "random
vendor catalog" support. Target: ≥85% per-cell, ≥90% per-card on both
holdouts with zero per-vendor configuration.

---

## 8. Cost

| Vendor | Pages | Cards | API spend |
|---|---:|---:|---:|
| Nike HO26 | 9 | 110 | **$0.53** |
| Adidas FW26 premium | 31 | 346 | **$1.73** |
| Converse HO26 | 17 | 15 | **$0.34** |

Average across these three: ~$0.87 per catalog.

**Projection at scale:**
- 100 catalogs/month: ~$80/month
- 500 catalogs/month: ~$400/month

Anthropic prompt-caching is enabled on system prompts (ephemeral cache
TTL = 5 min). Within-catalog calls reuse cached system prompts after the
first page.

Each pipeline run writes a sidecar (`<doc>.v2.cards.json`) so re-runs
after prompt tweaks or write.py changes are free (no API calls).

---

## 9. Known limitations (v1) & v2 roadmap

### Image binding deferred to v2

Column A in the output xlsx is intentionally empty. Per-card VLM photo
extraction was evaluated and reached only ~50% binding accuracy on dense
multi-section grids — not viable for production. Three architectural
approaches remain unexplored and are scheduled for v2:

1. **Native PDF XObject extraction + nearest-neighbor SKU binding**
   ($0 cost, ~1 day work, highest confidence for vector PDFs)
   - PyMuPDF exposes embedded image XObjects with pixel positions
   - For each SKU's PyMuPDF text anchor, bind nearest XObject above
   - The v0 cascade tried this but with buggy binding logic; we have
     clean anchors now
2. **Meta SAM (Segment Anything Model)** (~1-2 days, ~$0 marginal cost
   after first model download)
   - Click position from PyMuPDF SKU anchor → exact silhouette mask
   - Works on rasterized image-only PDFs where XObjects don't help
3. **Hybrid**: XObject first (fast path), SAM fallback (image-only pages)

### Other v1 caveats

- **Image-only PDFs** (Converse pages 5-7, 9) extract correctly via VLM
  but can't be verified against text layer; flagged as `vlm_only_no_region`.
- **ssg vocab gaps**: `description_map.json` was built from v0 vendor runs;
  new HO26 / FW26 descriptions aren't in it. ssg drops to ~33% "passing"
  on new descriptions — but these are not "wrong", just unverified.
  Closing this is a vocab-enrichment task.
- **`extract_single_card` retry costs**: dense grid pages (Nike page 1
  with 38 cards) trigger 30+ per-card retry calls, doubling per-page cost.
  Acceptable today (~$1.73 total for Adidas's 31 pages); could be
  optimized via tighter row-context geometry in future.

### Scaling concerns

- **No batch API** yet. If Kith starts processing 1000+ catalogs/month,
  switching to Anthropic's batched messages API would cut latency and
  cost ~50%.
- **No caching of card detection across versions**. Today every prompt
  tweak invalidates the full sidecar (it's monolithic JSON). Would benefit
  from per-phase caching when iterating.

---

## 10. Project layout

```
KithxKeeloShoeBuying/
├── README.md                       # quick start + accuracy snapshot
├── ARCHITECTURE.md                 # this file
├── LICENSE
├── BUYSHEET_template.xlsx          # immutable output template
├── apps_script/                    # GS image-in-cell helper (v2-relevant)
└── buysheet_v2/
    ├── pyproject.toml
    ├── cli.py                      # python -m buysheet_v2 entry point
    ├── pipeline.py                 # orchestrator
    ├── ingest.py                   # PDF -> renders + text
    ├── classify.py                 # Opus 4.7 layout classification
    ├── cards.py                    # Sonnet 4.6 card detection
    ├── extract.py                  # Sonnet 4.6 structured extraction + retry
    ├── verify.py                   # semantic oracle
    ├── consistency.py              # SKU dedup + brand voting
    ├── confidence.py               # per-cell confidence model
    ├── write.py                    # xlsx generation
    ├── schemas/                    # Pydantic models
    ├── prompts/                    # versioned VLM prompts
    ├── vocab/                      # closed vocabs (color, mg/sg/ssg, etc)
    ├── lifted/                     # helpers preserved from v0 (page render, vocab norm)
    └── tests/
        ├── eval_harness.py         # per-vendor accuracy reporter
        ├── scaffold_golden.py      # build ground-truth templates
        ├── golden/                 # hand-verified Nike, Adidas, Converse
        └── holdout/                # Hoka, Mizuno — cold-vendor ship gate
```

---

## 11. Decisions log

| Decision | Why |
|---|---|
| Sonnet 4.6 (not Opus) for per-page extraction | 5x cheaper, accuracy parity on layout tasks |
| Opus 4.7 only for doc-level classify | Single call per catalog; higher reasoning quality justified |
| Render @ 1568px long edge | Matches Anthropic vision input ceiling; returned bboxes in known coord system |
| Structured Outputs (vs. JSON-mode + parse) | Guarantees schema conformance; zero retry loops |
| Per-card retry fallback (vs. one-shot per-page) | Dense grids cause early-stop end_turn; retry recovers them at small extra cost |
| Semantic oracle (vs. structural scorer) | v0's structural scorer shipped wrong xlsx; oracle catches semantic errors |
| Skip image embedding in v1 | Per-card VLM photo extraction was ~50% accurate; not viable; v2 approach identified |
| Multi-brand split (vs. single tab with brand column) | Template assumes 1 brand per workbook; per-tab preserves template contract |
| temperature=0 NOT settable (Claude 4.x) | API rejects; mitigated by per-card retry mechanism |
| Skip-on-existing sidecar caching | Re-runs after prompt iteration cost $0 |
