# KithxKeeloShoeBuying

VLM-first extraction pipeline that turns shoebuyer catalog PDFs into the
Kith buy-sheet template (`BUYSHEET_<vendor>.xlsx`) with per-cell confidence
scoring and a deterministic source-text verification oracle.

**Current release: text-only v1.** Field extraction (SKU, brand, description,
color, MG/SG/SSG, intro date, USD cost/retail) is wired through. Image
embedding in column A is intentionally deferred — see
[ARCHITECTURE.md](ARCHITECTURE.md) for the rationale and roadmap.

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
4. Final post: summary (cards extracted, % verified, amber + red cell counts,
   cost) and the finished `BUYSHEET_<vendor>_v2.xlsx` attached to the same
   thread.

Every PDF run is persisted to `~/buysheet_runs/<pdf_stem>/<timestamp>/` with
the source PDF, extraction sidecar, and output workbook — so post-hoc
accuracy analysis is always possible.

See [`.env.example`](.env.example) for the full Slack app setup checklist.

## What it does

```
PDF
 → ingest         render pages @ 1568px long edge + extract text layer
 → classify       Opus 4.7, 1 call: layout type, multi-brand?, expected fields
 → cards          Sonnet 4.6 vision per page: card bboxes + SKU hints
 → extract        Sonnet 4.6 + Structured Outputs: full ProductCard per page
                  (with per-card retry fallback for dense grids)
 → verify         deterministic semantic oracle (every value must appear in
                  source text within the SKU's card region)
 → consistency    SKU dedup + brand voting + multi-brand workbook split
 → write          BUYSHEET-template xlsx: 1 tab per brand, amber + cell
                  comments for low-confidence cells
```

Detailed walkthrough in [ARCHITECTURE.md](ARCHITECTURE.md).

## Layout

```
KithxKeeloShoeBuying/
├── README.md                       # this file
├── ARCHITECTURE.md                 # CTO-facing system overview
├── LICENSE
├── .env.example
├── .gitignore
├── BUYSHEET_template.xlsx          # output template (immutable shared asset)
├── apps_script/
│   ├── INSTALL.md                  # one-time Google Sheets setup
│   └── fix_images.gs               # GS image-in-cell helper (kept for future image-binding release)
└── buysheet_v2/                    # main pipeline package
    ├── pyproject.toml
    ├── cli.py                      # python -m buysheet_v2 {run,eval,debug-cards,cost}
    ├── pipeline.py                 # orchestrator
    ├── ingest.py                   # PDF -> renders + text
    ├── classify.py                 # doc-level layout classification (Opus 4.7)
    ├── cards.py                    # per-page card detection (Sonnet 4.6)
    ├── extract.py                  # per-card structured extraction (+ retry)
    ├── verify.py                   # semantic oracle (10 field-level verifiers)
    ├── consistency.py              # SKU dedup, brand voting, multi-brand split
    ├── confidence.py               # per-cell confidence scoring
    ├── write.py                    # xlsx generation
    ├── schemas/                    # Pydantic models (ProductCard, etc)
    ├── prompts/                    # versioned VLM prompts
    ├── vocab/                      # closed vocabularies (color synonyms, MG/SG/SSG, etc)
    ├── lifted/                     # helpers reused from v1 (page render, vocab norm)
    └── tests/
        ├── eval_harness.py         # per-vendor accuracy reporter
        ├── scaffold_golden.py      # build ground-truth templates from current outputs
        ├── golden/                 # hand-verified ground truth (Nike, Adidas, Converse)
        └── holdout/                # vendors NEVER seen during dev — cold-vendor ship gate
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

Per-catalog API spend (Sonnet 4.6 + Opus 4.7):

| Vendor | Pages | API cost |
|--------|------:|---------:|
| Nike HO26 | 9 | $0.53 |
| Adidas FW26 premium | 31 | $1.73 |
| Converse HO26 | 17 | $0.34 |

100 catalogs/month at this rate: ~$80/month. Sidecar caching means re-runs
are free.

## Known limitations (v1)

- **Column A (PHOTO) is intentionally empty.** Native VLM-driven photo
  extraction was tried and reached only ~50% binding accuracy on dense
  multi-section grids. The right approach — native PDF XObject extraction
  + nearest-neighbor SKU binding (free) or Meta SAM-based segmentation —
  is scheduled for v2. See [ARCHITECTURE.md](ARCHITECTURE.md#image-binding-deferred-to-v2).
- **Image-only catalogs** (rasterized PPT exports) extract correctly via
  vision but can't be verified against a text layer. Confidence reports
  these as `vlm_only_no_region` rather than confirmed.
- **mg / sg / ssg vocab coverage gaps** show up as "uncertain" rather than
  wrong. Closing them is a `vocab/description_map.json` enrichment task,
  not a model issue.

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
