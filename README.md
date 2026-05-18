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

# 3. Run on a PDF
python -m buysheet_v2 run path/to/<vendor>.pdf --vendor-key <vendor>

# Output: BUYSHEET_<vendor>_v2.xlsx at repo root
```

The first run on a PDF takes 3-15 minutes (Sonnet 4.6 vision calls per page);
subsequent runs are cheap because per-doc extraction is cached in a sidecar.

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

After-mg-relaxation oracle scoring, per-cell:

| Vendor | Cards | All-fields | Contradicted | Core fields* |
|--------|------:|-----------:|-------------:|-------------:|
| Nike HO26 | 110 | **87.4%** | 3.2% | **95.0%** |
| Adidas FW26 premium | 346 | **87.4%** | 1.6% | **93.5%** |
| Converse HO26 | 15 | 69.2% | 6.3% | 69.6% |

\* Core fields = sku, description, color, brand, usd_cost, intro_date,
standard_color (excludes mg/sg/ssg which depend on vocab coverage)

Converse is the harder vendor because 7 of 17 pages are image-only
PPT-exported pages (no text layer); the semantic oracle can't verify those
values against source text, so they're left as VLM-only confidence.

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
