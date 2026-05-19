# KithxKeeloShoeBuying

VLM-first extraction pipeline that turns shoebuyer catalog PDFs into the Kith
buy-sheet template — text fields, product photos in column A, and a side-by-
side Google Sheet ready for the buying team to use in <30 minutes.

**Current release: v2 (text + photos + Sheets delivery), shipped 2026-05-19.**

What's wired through:
- Field extraction (SKU, brand, description, color, MG/SG/SSG, intro date,
  USD cost/retail) via Claude Sonnet 4.6 + Anthropic Structured Outputs.
- Per-cell semantic verification oracle (every claimed value must appear in
  source page text within its card region; SKU misreads get blank-and-flag
  treatment).
- **Photo binding** via YOLO-World shoe detection + Sonnet matcher
  (decouples *where are shoes?* from *which SKU is which?* — works on PPTX
  grids, lookbooks with description-as-SKU, hero+swatch layouts).
- **Path C Google Sheets API direct delivery** — Slack bot posts a
  ready-to-use Sheet URL alongside the xlsx, photos embedded in-cell from
  sheet-creation time (no manual Apps Script step, no floating-image drift).
- Honest Slack summary: per-field fill rates, page failure count, row-
  truncation warnings, embedded photo count.

Full architecture + accuracy benchmarks + decisions log in
[ARCHITECTURE.md](ARCHITECTURE.md).

---

## Quick start

```bash
# 1. Install
pip install -e ./buysheet_v2

# 2. Set your API key
cp .env.example .env
# edit .env, set ANTHROPIC_API_KEY=sk-ant-...

# 3. Run on a PDF (CLI)
python -m buysheet_v2 run path/to/<vendor>.pdf --vendor-key <vendor>

# Output: BUYSHEET_<vendor>_v2.xlsx at repo root
```

The first run on a PDF takes 3-15 minutes (Sonnet 4.6 vision calls per page);
subsequent runs are cheap because per-doc extraction is cached in a sidecar.

### Slack-bot mode (preferred for buyers)

The agent ships with a Socket Mode Slack bot so a buyer can drop a PDF into a
channel and get the finished workbook back without touching the CLI:

```bash
# After filling in SLACK_BOT_TOKEN, SLACK_APP_TOKEN, SLACK_CHANNEL_ID in .env
python -m buysheet_v2.slack_bot
```

User flow:

1. Drop PDF in the configured channel.
2. Bot replies in thread with an ETA (~20 s/page).
3. Pings at 25 / 50 / 75% during extraction.
4. Final post: honest summary (cards extracted with page-failure count, % of
   cells verified, amber/red counts, **per-field fill rate breakdown**,
   embedded photo count, truncation warning if catalog exceeds template
   row cap, cost) followed by both the finished `BUYSHEET_<vendor>_v2.xlsx`
   AND the Google Sheet URL ready to open.

Every PDF run is persisted to `~/buysheet_runs/<pdf_stem>/<timestamp>/` with
the source PDF, extraction sidecar, and output workbook — so post-hoc
accuracy analysis is always possible.

See [`.env.example`](.env.example) for the full Slack app setup checklist.

## What it does

```
PDF (uploaded to Slack)
 → ingest         render pages @ 1568px long edge + extract text layer
 → classify       Opus 4.7, 1 call: layout type, multi-brand?, expected fields
 → yolo_detect    YOLO-World detects all shoes per page (free, local) +
                  density picker (Sonnet) chooses imgsz 1280/1920/2560 +
                  annotated.png with red numbered boxes for the matcher
 → cards          Sonnet 4.6 vision per page: card bboxes + SKU hints
 → extract        Sonnet 4.6 + Structured Outputs: full ProductCard per page
                  (with per-card retry fallback for dense grids)
 → photo_match    Sonnet matches each SKU to a numbered YOLO box on the
                  annotated page — overwrites card.photo_bbox_px with the
                  YOLO bbox (layout-agnostic, no text-anchor dependency)
 → verify         deterministic semantic oracle (every value must appear in
                  source text within the SKU's card region) + SKU
                  blank-and-flag defense for single-char vision misreads
 → consistency    SKU dedup + brand voting + multi-brand workbook split
 ├→ write         BUYSHEET-template xlsx: 1 tab per brand, photos in col A,
 │                amber/red tier formatting + cell comments
 └→ sheets_writer Google Sheets API direct delivery: photos in-cell at
                  sheet creation (no manual Apps Script, no drift),
                  tier formatting + cell notes + domain sharing
```

Detailed walkthrough + per-vendor accuracy benchmarks + decisions log in
[ARCHITECTURE.md](ARCHITECTURE.md).

## Layout

```
KithxKeeloShoeBuying/
├── README.md                       # this file
├── ARCHITECTURE.md                 # CTO-facing system overview (full detail)
├── ACCURACY_EXPECTATIONS.md        # per-vendor fill-rate expectations
├── LICENSE
├── .env.example
├── .gitignore
├── BUYSHEET_template.xlsx          # output template (immutable shared asset)
├── apps_script/                    # legacy "Fix images" Apps Script
│                                   # (superseded by Path C direct Sheets API,
│                                   #  kept as fallback for xlsx-only workflows)
└── buysheet_v2/                    # main pipeline package
    ├── pyproject.toml
    ├── cli.py                      # python -m buysheet_v2 entry point
    ├── pipeline.py                 # orchestrator (YOLO+matcher integrated)
    ├── ingest.py                   # PDF → renders + text
    ├── classify.py                 # Opus 4.7 layout classification
    ├── yolo_detect.py              # YOLO-World shoe detection pre-pass
    ├── photo_match.py              # density picker + SKU→bbox matcher
    ├── photo_vlm.py                # LEGACY text-anchored bbox (graceful fallback)
    ├── phototune.py                # deterministic geometric photo fallback
    ├── cards.py                    # per-page card detection (Sonnet 4.6)
    ├── extract.py                  # per-card structured extraction + retry
    ├── verify.py                   # semantic oracle + SKU blank-and-flag
    ├── consistency.py              # SKU dedup, brand voting, multi-brand
    ├── confidence.py               # per-cell confidence scoring
    ├── write.py                    # xlsx generation with photos
    ├── sheets_writer.py            # Path C Google Sheets API delivery
    ├── schemas/                    # Pydantic models
    ├── prompts/                    # versioned VLM prompts
    ├── vocab/                      # closed vocabularies (color, MG/SG/SSG, etc)
    ├── models/                     # YOLO weights (auto-downloaded, gitignored)
    ├── lifted/                     # v0 helpers (page render, vocab norm)
    ├── slack_bot/                  # Socket Mode entrypoint + FIFO worker
    └── tools/
        ├── eval_against_goldens.py # per-vendor accuracy vs verified ground truth
        ├── multi_model_extract.py  # 3-way Sonnet+Opus+Gemini extraction
        ├── three_way_compare.py    # cross-model agreement classifier
        ├── scaffold_golden_v2.py + verify_golden.py
        ├── enrich_description_map.py + enrich_color_synonyms.py
        ├── golden/                 # hand-verified ground truth (6 vendors)
        └── holdout/                # vendors NEVER seen during dev — ship gate
```

## Verified accuracy (v1)

Per-cell oracle scoring, locked in via `tests/baseline_accuracy.json`:

| Vendor | Cards | Per-cell | Contradicted (= blank in xlsx) |
|--------|------:|---------:|-------------------------------:|
| Nike HO26 | 110 | **97.67%** | 0.20% |
| Adidas FW26 premium | 346 | **94.74%** | 2.03% |
| Converse HO26 | 15 | 74.83% | 6.29% |

Cold-vendor extractions seen during multi-PDF Slack testing (zero
configuration, auto-vocab-enriched):

| Vendor | Cards | Per-cell | Notes |
|--------|------:|---------:|-------|
| SPS 2024 (multi-brand) | 257 | ~96% | Brand 100%, color 92%, description 92% |
| Hoka SP27 Pinnacle (lookbook) | 172 | ~92% | Description 97.7%, intro_date 98.2% |

Converse is the harder vendor because 7 of 17 pages are image-only
PPT-exported pages (no text layer); the semantic oracle can't verify those
values against source text, so they're left as VLM-only confidence.

The 2026-05 oracle pass added four verifier fixes that lifted Hoka description
from 35% → 97.7%, Hoka intro_date from 0% → 98.2%, and SPS brand from 91% →
100%, all without re-extracting (the oracle changes alone closed the gap):
ligature folding (PDF `ﬀ`/`ﬁ` ↔ ASCII `ff`/`fi`), numeric date pattern
matching (`MM/DD` → `JAN`), page-level brand fallback for multi-brand catalogs,
and shared-description recognition for sibling-SKU sections.

Full per-field breakdown and per-catalog-type confidence tiers in
[ACCURACY_EXPECTATIONS.md](ACCURACY_EXPECTATIONS.md). Detailed pipeline
walkthrough in [ARCHITECTURE.md](ARCHITECTURE.md).

## Cost

Per-catalog API spend (Sonnet 4.6 extract + Opus 4.7 classify + Sonnet density
picker + Sonnet matcher + free local YOLO):

| Vendor | Pages | Cards | API cost | Wall time |
|--------|------:|------:|---------:|----------:|
| Nike HO26 | 9 | 110 | ~$1.20 | ~4 min |
| Hoka SP27 | 9 | 176 | ~$1.50 | ~5 min |
| Mizuno SS27 | 90 | 120 | ~$3.00 | ~10 min |
| SPS 2024 | 66 | 257 | ~$3.50 | ~11 min |
| Adidas FW26 premium | 31 | 343 | ~$4.50 | ~12 min |

Average across 5 validated catalogs: **~$2.74 per catalog**. At 100
catalogs/month: ~$275/month. Sidecar caching means re-runs are free.

## Photo binding (the v2 win)

The v1 text-anchored photo bbox strategy failed on catalogs where photos
aren't reliably near SKU text. v2 replaces it with YOLO-World detection +
Sonnet matcher (decouples *where are shoes?* from *which SKU is which?*):

| Catalog | Layout | v1 (text-anchored) | v2 (YOLO + matcher) |
|---|---|---:|---:|
| Hoka SP27 | Lookbook + sibling colorways | visible misalignment | **172/176 (98%)** |
| Nike HO26 PPTX | Tiny shoes + dense text | text-crops, not shoes | 91/111 (82%) |
| Salomon SS27 | Hero + swatch grid | mixed text/swatch crops | 140/150 (93%) |
| Mizuno SS27 | Description-as-SKU lookbook | **6/120** anchors | **120/120 (100%)** |
| SPS 2024 | Matrix table | not benchmarked | 256/257 (99.6%) |
| LLT (image-only) | Scanned PDF | (whole-card fallback) | 5 YOLO + 8 phototune |

YOLO weights (`yolov8s-worldv2.pt`, ~25MB) auto-download on first run.
`ultralytics` is an optional dep — when missing, the pipeline falls back
to the legacy text-anchored bbox path.

## Known limitations

- **Image-only catalogs** extract correctly via vision but text-field
  verification falls back to `vlm_only_no_region` confidence. YOLO+matcher
  still works on these when shoes are visible.
- **mg / sg / ssg vocab coverage gaps** show up as "uncertain" rather than
  wrong. Closing them is a `vocab/description_map.json` enrichment task.
- **YOLO inference is serial** (ultralytics.predict not thread-safe).
  Adidas-scale catalogs spend ~5 min in the YOLO pre-pass alone.

## Random / cold catalog onboarding

For a vendor we've never run before, the pipeline auto-enriches the vocab
cache as part of the first run — no per-vendor configuration needed:

```bash
python -m buysheet_v2 run files/<new_vendor>/<catalog>.pdf --vendor-key <new_vendor>
```

What happens on a cold run:

1. **ingest + classify + cards + extract** — Sonnet 4.6 / Opus 4.7 do the
   layout-agnostic vision work. Cold accuracy estimated 85-92% per cell.
2. **auto-enrich** — any product descriptions in this catalog that aren't
   yet in `vocab/description_map.json` get classified via one batched
   Claude call (~$0.03 per 30 novel descriptions). Persisted to repo;
   benefits every future catalog with shared silhouettes.
3. **semantic oracle** — runs with the freshly-enriched vocab. Final
   confidence reflects the enriched state.
4. **write** — `BUYSHEET_<vendor>_v2.xlsx` with amber-flagged uncertain cells.

To skip enrichment (e.g. for repeated test runs you don't want to grow
the cache):

```bash
# Programmatic API only:
from buysheet_v2.pipeline import run_pipeline
run_pipeline(pdf, auto_enrich=False)
```

## Accuracy validation via VLM-as-judge (Opus reviews Sonnet)

For independent verification of any extraction without re-running Sonnet,
run the judge tool against a cached sidecar. It calls Opus 4.7 in parallel
over the same PDF, compares per-card per-field values, and reports the
agreement rate plus sample disagreements.

```bash
# Estimate cost first (free, no API calls)
python -m buysheet_v2.tools.judge_existing --estimate path/to/<doc>.v2.cards.json

# Run the judge (Opus is ~5x Sonnet cost; expect ~$0.20-0.25 per page)
python -m buysheet_v2.tools.judge_existing path/to/<doc>.v2.cards.json

# Write the judge_agreement scores back into the sidecar for downstream use
python -m buysheet_v2.tools.judge_existing --update path/to/<doc>.v2.cards.json
```

Per-field disagreement counts surface real errors. On a Nike HO26 PPTX
sample (9 pages, 110 cards, $1.24 in Opus cost), the smoke test caught a
description-rotation mis-attribution between three adjacent SKUs that
neither model would have flagged alone — exactly the cross-card mistake
class the oracle is too lenient about.

Judge can also run inside the main pipeline (`run_pipeline(pdf, run_judge=True)`)
or by setting the bot to validate every upload. Off by default because
of the Opus cost surcharge.

## Ground-truth accuracy (the gold standard)

The metrics above (oracle pass rate, judge agreement) are strong proxies but
not the same as **true accuracy** — neither asks "does the extracted value
match what a human expert would write?" For that, we use hand-verified
goldens in `tests/golden/<vendor>.json`.

Workflow:

```bash
# 1. Scaffold a golden from the latest extraction (one-time per vendor)
python -m buysheet_v2.tools.scaffold_golden_v2 \
    ~/buysheet_runs/<vendor>/<timestamp>/<doc>.v2.cards.json \
    --vendor-key <vendor> --vendor-type athletic-grid --sample 25

# 2. Walk through each SKU and verify against the source PDF
python -m buysheet_v2.tools.verify_golden \
    buysheet_v2/tests/golden/<vendor>.json
#   For each SKU: 'a' to accept, 'e' to edit, 's' to skip, 'q' to quit.
#   ~30 seconds per SKU on a clean grid catalog; budget 30 min per vendor.

# 3. Run accuracy against verified ground truth
python -m buysheet_v2.tools.eval_against_goldens
#   Shows per-vendor + per-field accuracy vs hand-verified values.
#   Compares to tests/golden_baseline.json; exits 1 if any vendor regressed.

# 4. Lock in baseline after a verified improvement
python -m buysheet_v2.tools.eval_against_goldens --update-baseline
```

The eval harness produces the **true accuracy number** — strictly stronger
than the oracle proxy. It only counts SKUs marked `_verified: true` in the
golden file, so you can scaffold + verify incrementally without breaking
the baseline.

Cold-vendor ship gate (Phase 4 of the original design): the same workflow on
`tests/holdout/*.json` produces the gate metric. Holdouts are catalogs the
pipeline has never seen during development.

## Regression checking

After ANY change to `verify.py`, `extract.py`, prompts, or vocab files,
run the regression check before pushing:

```bash
python -m buysheet_v2.tools.verify_all
```

This re-runs the semantic oracle on every cached vendor sidecar, compares
to `tests/baseline_accuracy.json`, and exits non-zero if any vendor
regressed >0.5pp. To lock in a new baseline after a verified improvement:

```bash
python -m buysheet_v2.tools.verify_all --update-baseline
```

## Run a vendor (specific commands)

```bash
# Default (full pipeline + per-cell confidence + sidecar caching)
python -m buysheet_v2 run files/<vendor>/<catalog>.pdf --vendor-key <vendor>

# Eval against a golden test set
python -m buysheet_v2 eval <vendor_key>

# Debug a single page's card detection (writes overlay PNG)
python -m buysheet_v2 debug-cards files/<vendor>/<catalog>.pdf --page 5
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design walkthrough.
