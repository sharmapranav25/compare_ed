# buysheet_v2 — VLM-First Card-Oracle Pipeline

Extracts product data from any shoebuyer catalog PDF into Kith's
`BUYSHEET_template.xlsx`, using Claude vision as the primary extraction
primitive and deterministic verification as the correctness gate.

## Why this exists

The v1 pipeline (in `../phase1/` through `../phase5/`) serializes PDFs to
flat text and rebinds extracted fields to SKUs via regex + cluster
heuristics. Empirically it reaches ~57-68% per-card semantic accuracy on
real Kith vendor catalogs even when the report card claims "100% fill,
1.5% flagged" — the correctness scorer measures structural noise, not
semantic correctness. Per-vendor regex tuning is a dead end; the next
new layout convention surfaces the next bug.

v2 replaces the read-side primitives entirely. The unit of truth is a
**product card** (a 2D region containing one product's photo + name + SKU
+ color + price), detected by Claude vision per page. Once cards are
bounded, the binding problem disappears — every field comes from inside
a known card region. A deterministic oracle verifies every extracted
value appears in the page text within the card bounds.

## Architecture

```
PDF
 → ingest      render every page to PNG @ 180 DPI + extract text layer
 → classify    Opus 4.7 reads first 3 pages → layout type, multi-brand?
 → cards       Sonnet 4.6 vision per page → card bbox list
 → extract     Sonnet 4.6 + Structured Outputs → List[ProductCard] per page
 → verify      semantic oracle: value-in-source-text-within-card-region
              + vocab lookups (color_synonyms, description_map)
 → consistency SKU dedup + per-brand voting + multi-brand tab split
 → write       BUYSHEET_<vendor>.xlsx, one tab per brand,
              photos cropped from card_bbox, amber + cell comments for
              <90% confidence cells
 → report      per-card confidence summary + manual review queue
```

## Folder layout

```
buysheet_v2/
├── README.md                # this file
├── pyproject.toml           # standalone package; portable to a separate repo
├── cli.py                   # python -m buysheet_v2 <subcommand>
├── pipeline.py              # orchestrator (Phase 2)
├── ingest.py                # PDF → renders + text (Phase 1)
├── classify.py              # layout classification (Phase 2)
├── cards.py                 # per-page card detection (Phase 1)
├── extract.py               # per-card field extraction (Phase 1)
├── verify.py                # semantic oracle (Phase 3)
├── consistency.py           # cross-card validation (Phase 3)
├── confidence.py            # per-cell confidence model (Phase 2)
├── write.py                 # workbook write (Phase 2)
├── schemas/                 # Pydantic models (Phase 0 — done)
│   ├── card.py              # ProductCard + CardBbox
│   ├── doc_layout.py        # LayoutClassification
│   └── extraction_result.py # PageExtraction + CatalogExtraction
├── prompts/                 # versioned VLM prompts (Phase 1)
│   ├── layout_classify.md
│   ├── card_detect.md
│   └── card_extract.md
├── vocab/                   # immutable references copied from ../phase3/
│   ├── color_synonyms.json  # 496 vendor color tokens → canonical
│   ├── buysheet_vocab.json  # template Product Data closed vocabularies
│   ├── silhouette_ssg_map.json
│   ├── section_vocab.json
│   ├── gender_vocab.json
│   └── description_map.json # 4000+ description → SG/SSG cached
├── lifted/                  # helpers copied from v1 (Phase 0 — done)
│   ├── pdf_render.py        # render_page_image + b64_image_block
│   ├── photo_embed.py       # crop_to_silo + embed_photo
│   └── vocab_normalize.py   # normalize_intro_date / vendor / season / gender
└── tests/
    ├── eval_harness.py      # run pipeline + compare to golden + report
    ├── golden/              # hand-verified ground truth per vendor (tune-against)
    └── holdout/             # vendors NEVER seen during tuning — the ship gate
```

## Usage

```bash
# Run full pipeline on a PDF
python -m buysheet_v2 run files/<vendor>/<catalog>.pdf [--vendor-key X]

# Debug: render a page with card bboxes overlaid
python -m buysheet_v2 debug-cards files/<vendor>/<catalog>.pdf --page 5

# Eval against golden ground truth
python -m buysheet_v2 eval <vendor_key>

# Cost breakdown
python -m buysheet_v2 cost files/<vendor>/<catalog>.pdf
```

## Ship gate (Phase 4)

A new vendor is "ready to ship" when, with zero per-vendor config:

- **Per-cell accuracy ≥85%** on `tests/holdout/<vendor>.json`
- **Per-card accuracy ≥90%** on the same holdout

`tests/holdout/` is written ONCE before any tuning and never touched
during development. If a release candidate fails the gate, iterate
`prompts/*.md` (not per-vendor code) until it passes. If three iterations
don't move the needle, scope down (fewer fields, restrict to single-brand)
rather than ship a regression.

## Confidence model

Every extracted (sku, field, value) gets a confidence in [0.0, 1.0]:

- **1.0** — value passed the semantic oracle (string appears in source page
  text within the card region)
- **0.7** — value derived from vocab lookup (color_synonyms,
  description_map) AND the input string appears in the source
- **0.5** — value from VLM only; no source confirmation
- **0.0** — value contradicts source (extracted SKU != text-search SKU
  in same region) — flagged for review, value HIDDEN from xlsx with raw
  preserved in cell comment

Cells with confidence <0.9 get amber background + cell comment with
provenance. The buyer reviews amber cells in the existing xlsx workflow;
no separate review app required for v1.

## What we explicitly do NOT do (v1)

- Pre-Size, PU1, REORDER, LAUNCH, EUR/GBP/CAD columns (schema-extension
  for v2; the architecture supports adding fields without rewrite)
- Web review UI (xlsx amber + comments instead)
- Self-improving vocab loop (color_synonyms is static in v1)
- Touch any v1 code (`../phase1/` through `../phase5/` are untouched)
- Modify `BUYSHEET_template.xlsx` (immutable shared asset)

## Lifted from v1 (what survives the architectural cull)

| v2 file | v1 source | What it does |
|---|---|---|
| `lifted/pdf_render.py` | `phase3/annotate.py` | PDF page → PNG bytes at consistent DPI |
| `lifted/photo_embed.py` | `phase5/populate.py` | Silhouette isolation + openpyxl image embed |
| `lifted/vocab_normalize.py` | `phase5/populate.py` | Date/vendor/season/gender normalization |
| `vocab/*.json` | `phase3/*.json` | Closed-vocabulary lookups (color, description, silhouette) |
| (referenced) | `BUYSHEET_template.xlsx` | The output template (immutable) |
| (referenced) | `apps_script/fix_images.gs` | GS in-cell image conversion (vendor-agnostic) |

## NOT lifted (architectural mistake)

- `phase1/parse.py` — DocLing is slow + fragments stylized text
- `phase1/reconcile.py` — redundant with image-direct extraction
- `phase1/normalize_skus.py` — per-doc SKU regex (VLM extracts directly)
- `phase2/extract_fields.py` — per-doc field regex (VLM returns full schema)
- `phase3/deterministic_pipeline.py` — flat-text rebinding (the bug source)
- `phase3/assign.py` — cluster repair (redundant with card-bounded extraction)
- `phase3/match_photos.py` — B1-B4 cascade (replaced by VLM photo_bbox)
- `phase3/layout_parser.py` — matrix-only fast path (VLM handles all uniformly)
- `phase5/populate.py` (most) — relational-shape consumer; we write directly
- `phase5/correctness_scorer.py` — structural-only flagger (semantic oracle replaces)

## Implementation phases

- **Phase 0** (done): folder, schemas, lifted helpers, eval harness skeleton
- **Phase 1**: single-page card detection + extraction MVP, smoke test
- **Phase 2**: full-document pipeline + write.py + confidence.py
- **Phase 3**: semantic oracle + multi-brand split + Hanger Clinic stress test
- **Phase 4** (THE GATE): cold accuracy on Hoka SP27 + Mizuno SS27 holdouts
- **Phase 5**: production hardening, caching, cost monitoring

Full plan: `/Users/edwardionel/.claude/plans/imperative-scribbling-dawn.md`
