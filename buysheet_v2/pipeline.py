"""End-to-end orchestrator: PDF in, CatalogExtraction out.

Phase 2 wiring — Phase 3 will add verify.py (semantic oracle) and
consistency.py (brand voting + multi-brand split) between extract and write.

Steps:
  1. ingest    PDF -> renders + text per page
  2. classify  Opus 4.7 on first 3 pages -> layout metadata
  3. cards     Sonnet 4.6 per page -> per-page card bboxes
  4. extract   Sonnet 4.6 + Structured Outputs per page -> ProductCard list
  5. confidence (stub in Phase 2; verify.py replaces in Phase 3)

Skip-on-existing: each phase's output is sidecar-cached so re-runs after
prompt tweaks only redo the changed phase.

Pricing constants — anthropic.com/pricing (Sonnet 4.6, Opus 4.7, May 2026):
  Sonnet 4.6: $3 / $15 per MTok (in / out)
  Opus 4.7:  $15 / $75 per MTok
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import anthropic

# Progress callback signature: (phase_name, pct_0_to_1, human_message) -> None.
# Phases emitted: "start", "ingest", "classify", "extract", "enrich",
# "verify", "done". Total wall-clock weight is dominated by per-page extract
# (~90%); classify/verify/enrich are quick.
ProgressCallback = Callable[[str, float, str], None]

from buysheet_v2.cards import detect_cards_on_page
from buysheet_v2.classify import classify_layout
from buysheet_v2.extract import extract_with_retry
from buysheet_v2.ingest import ingest
from buysheet_v2.schemas.extraction_result import (
    CatalogExtraction, PageExtraction,
)
from buysheet_v2.verify import oracle_summary, verify_catalog
from buysheet_v2 import yolo_detect
from buysheet_v2.photo_match import match_skus_to_bboxes, pick_yolo_density

SONNET_IN_PER_MTOK = 3.0
SONNET_OUT_PER_MTOK = 15.0
OPUS_IN_PER_MTOK = 15.0
OPUS_OUT_PER_MTOK = 75.0


def _cost_sonnet(in_tok: int, out_tok: int, cache_read: int = 0) -> float:
    # Cache reads bill at ~10% of input rate (approximation)
    return (
        (in_tok / 1_000_000) * SONNET_IN_PER_MTOK
        + (out_tok / 1_000_000) * SONNET_OUT_PER_MTOK
        + (cache_read / 1_000_000) * SONNET_IN_PER_MTOK * 0.1
    )


def _cost_opus(in_tok: int, out_tok: int, cache_read: int = 0) -> float:
    return (
        (in_tok / 1_000_000) * OPUS_IN_PER_MTOK
        + (out_tok / 1_000_000) * OPUS_OUT_PER_MTOK
        + (cache_read / 1_000_000) * OPUS_IN_PER_MTOK * 0.1
    )


def run_pipeline(
    pdf_path: Path,
    *,
    vendor_key: Optional[str] = None,
    client: Optional[anthropic.Anthropic] = None,
    cache_sidecar: bool = True,
    auto_enrich: bool = True,
    verbose: bool = True,
    progress_callback: Optional[ProgressCallback] = None,
    run_judge: bool = False,
) -> CatalogExtraction:
    """Run the full VLM-first pipeline on a PDF and return a CatalogExtraction.

    The result is also written to <pdf_stem>.cards.json next to the PDF
    (so re-runs can pick up cached cards before write.py).

    progress_callback: optional hook invoked at key phases as
    (phase, pct, message). The Slack bot uses this to relay 25/50/75/100%
    pings to the user; pct is monotonically non-decreasing in [0.0, 1.0].
    """
    pdf_path = pdf_path.resolve()
    if vendor_key is None:
        vendor_key = pdf_path.stem.lower().replace(" ", "_")
    if client is None:
        client = anthropic.Anthropic()

    def _emit(phase: str, pct: float, msg: str) -> None:
        if progress_callback is not None:
            try:
                progress_callback(phase, pct, msg)
            except Exception as e:
                if verbose:
                    print(f"[pipeline] progress_callback error ({phase}): "
                          f"{type(e).__name__}: {e}")

    sidecar = pdf_path.with_suffix("").with_name(f"{pdf_path.stem}.v2.cards.json")
    if cache_sidecar and sidecar.exists():
        if verbose:
            print(f"[pipeline] loading cached extraction from {sidecar.name}")
        _emit("cached", 1.0, f"Cached extraction loaded from {sidecar.name}")
        return CatalogExtraction.model_validate_json(sidecar.read_text())

    _emit("start", 0.0, f"Starting extraction of {pdf_path.name}")

    if verbose:
        print(f"[pipeline] ingesting {pdf_path.name}...")
    doc = ingest(pdf_path)
    if verbose:
        print(f"  -> {doc.page_count} pages rendered + text-extracted")
    _emit("ingest", 0.02, f"Parsed {doc.page_count} pages from {pdf_path.name}")

    result = CatalogExtraction(pdf_path=str(pdf_path), vendor_key=vendor_key)
    # Per-page matcher outcome markers, attached to CardConfidence.flags after
    # verify_catalog runs. Values: "matcher_failed" (matcher raised),
    # "no_match" (matcher returned but assigned no boxes). Pages where the
    # matcher worked normally aren't in this dict.
    page_match_markers: dict[int, str] = {}

    # Step 1: doc-level classify (Opus, 1 call)
    if verbose:
        print(f"[pipeline] classifying layout (Opus 4.7, first 3 pages)...")
    layout, c_usage = classify_layout(doc, client=client)
    result.layout = layout
    result.tokens_input += c_usage["input_tokens"]
    result.tokens_output += c_usage["output_tokens"]
    result.tokens_cache_read += c_usage["cache_read_tokens"]
    result.cost_usd += _cost_opus(
        c_usage["input_tokens"], c_usage["output_tokens"], c_usage["cache_read_tokens"],
    )
    if verbose:
        print(f"  -> layout={layout.layout_type}  multi_brand={layout.is_multi_brand}  "
              f"expected_fields={layout.expected_fields_present}")
    _emit("classify", 0.05,
          f"Layout: {layout.layout_type}"
          + (" · multi-brand catalog" if layout.is_multi_brand else " · single-brand"))

    # Step 1.5: YOLO pre-pass (serial, ~5-15s/page). Detects every shoe on
    # every page + renders an annotated PNG with red numbered boxes for the
    # matcher to use. Caches to <pdf>.v2.yolo.json so re-runs cost $0 (only
    # the Sonnet density picker re-runs without cache, ~$0.005/page).
    yolo_detections: dict[int, dict] = {}
    if yolo_detect.is_available():
        if verbose:
            print(f"[pipeline] YOLO pre-pass on {doc.page_count} pages...")
        density_picker = lambda p: pick_yolo_density(p, client=client)  # noqa: E731
        # Open a pdfium handle for the hi-res matcher annotation path.
        # render_page_for_matcher needs the source PDF (not just the
        # 1568-px IngestedPage.png_bytes) so dense grid captions are
        # legible. Hi-res coords are scaled and consumed entirely inside
        # annotate_page_at_scale; the bbox lookup table downstream consumers
        # read stays in 1568-px space.
        pdfium_for_yolo = None
        try:
            import pypdfium2 as _pdfium
            pdfium_for_yolo = _pdfium.PdfDocument(str(pdf_path))
        except Exception as e:
            if verbose:
                print(f"  -> couldn't open pdfium handle ({type(e).__name__}: {e}) — "
                      f"matcher will see 1568-px annotated pages")
        try:
            yolo_detections = yolo_detect.run_detect_prepass(
                doc.pages, sidecar_dir=pdf_path.parent,
                pdf_stem=pdf_path.stem, density_picker=density_picker,
                pdf=pdfium_for_yolo,
            )
            total_shoes = sum(len(d["bboxes"]) for d in yolo_detections.values())
            if verbose:
                print(f"  -> detected {total_shoes} shoes across "
                      f"{len(yolo_detections)} pages")
        except Exception as e:
            if verbose:
                print(f"  -> YOLO pre-pass FAILED ({type(e).__name__}: {e}) — "
                      f"falling back to legacy photo_vlm path")
            yolo_detections = {}
        finally:
            if pdfium_for_yolo is not None:
                try:
                    pdfium_for_yolo.close()
                except Exception:  # noqa: BLE001
                    pass
        _emit("detect", 0.07,
              f"Detected shoes on {len(yolo_detections)} pages via YOLO")

    # Step 2/3: per-page card detection + extraction (Sonnet, 2 calls per page)
    # Per-page work is the dominant wall-clock cost; map it into [0.05, 0.95]
    # so the bot's 25/50/75 thresholds land naturally inside the extract loop.
    EXTRACT_START_PCT = 0.05
    EXTRACT_END_PCT = 0.95
    total_pages = max(1, doc.page_count)
    cards_so_far = 0
    for page in doc.pages:
        if verbose:
            print(f"[pipeline] page {page.page_no}/{doc.page_count}  "
                  f"(text={len(page.text)}ch)", end="  ")
        try:
            cards, d_usage = detect_cards_on_page(
                page, client=client,
                layout_type=layout.layout_type,
                is_multi_brand=layout.is_multi_brand,
            )
            result.tokens_input += d_usage["input_tokens"]
            result.tokens_output += d_usage["output_tokens"]
            result.tokens_cache_read += d_usage["cache_read_tokens"]
            result.cost_usd += _cost_sonnet(
                d_usage["input_tokens"], d_usage["output_tokens"], d_usage["cache_read_tokens"],
            )

            extracted, e_usage = extract_with_retry(
                page, cards, client=client,
                layout_type=layout.layout_type,
                is_multi_brand=layout.is_multi_brand,
                expected_fields=layout.expected_fields_present,
                catalog_brand=(
                    None if layout.is_multi_brand
                    else (layout.notes or "").split(":")[0].strip() or None
                ),
            )
            result.tokens_input += e_usage["input_tokens"]
            result.tokens_output += e_usage["output_tokens"]
            result.tokens_cache_read += e_usage["cache_read_tokens"]
            result.cost_usd += _cost_sonnet(
                e_usage["input_tokens"], e_usage["output_tokens"], e_usage["cache_read_tokens"],
            )

            # YOLO photo-bbox matcher: if we have YOLO detections for this
            # page, ask Sonnet to match each extracted SKU to its numbered box
            # on the annotated page. Overwrite card.photo_bbox_px with the
            # matched bbox (which is much tighter + correctly-anchored than
            # the VLM's own photo_bbox_px from extract.py).
            #
            # Text-layer pre-filter: before sending the SKU list to the
            # matcher, drop SKUs not physically present in the PDF's text
            # layer. The matcher's first-SKU-wins de-dup
            # (photo_match.py:243-256) is vulnerable to phantom SKUs
            # claiming real products' numbered boxes — the pre-filter
            # closes that gap. On image-only PDFs (text_layer_present=False)
            # the full list is passed through unchanged.
            page_det = yolo_detections.get(page.page_no, {})
            yolo_bboxes = page_det.get("bboxes") or []
            annotated_path = page_det.get("annotated_path")
            matched_count = 0
            if yolo_bboxes and annotated_path and extracted:
                from buysheet_v2.sku_text_check import present_skus_in_text
                present, text_layer_present = present_skus_in_text(
                    page.text, [c.sku for c in extracted],
                )
                if text_layer_present:
                    matcher_input = [c for c in extracted if c.sku in present]
                    n_phantom = len(extracted) - len(matcher_input)
                else:
                    matcher_input = extracted
                    n_phantom = 0
                page_marker: Optional[str] = None
                if not matcher_input:
                    # All extracted SKUs were filtered out as phantoms — the
                    # matcher never gets called, but the buyer's expectation
                    # was "this page had YOLO detections + extracted cards"
                    # so flag it the same as "matcher returned nothing."
                    page_marker = "no_match"
                else:
                    try:
                        matched = match_skus_to_bboxes(
                            Path(annotated_path), matcher_input, yolo_bboxes,
                            client=client,
                        )
                        if not matched:
                            # Matcher returned an empty assignment dict
                            page_marker = "no_match"
                        for card in extracted:
                            bbox = matched.get(card.sku)
                            if bbox:
                                card.photo_bbox_px = list(bbox)
                                matched_count += 1
                    except Exception as e:
                        page_marker = "matcher_failed"
                        if verbose:
                            print(f" matcher_FAILED({type(e).__name__})", end="")
                if page_marker is not None:
                    page_match_markers[page.page_no] = page_marker

            page_result = PageExtraction(
                page=page.page_no, cards=extracted, page_text=page.text,
            )
            result.pages.append(page_result)
            cards_so_far += len(extracted)

            # Persist this page's sidecar (in addition to the catalog-level
            # one written at end-of-run). Lets a crash mid-extract leave
            # partial state on disk that C6's backfill CLI can recover.
            # Best-effort: a write failure here doesn't fail the page.
            try:
                from buysheet_v2.sidecar import save_page_sidecar
                save_page_sidecar(pdf_path, page.page_no, page_result)
            except Exception as _e:  # noqa: BLE001
                if verbose:
                    print(f" sidecar_save_FAILED({type(_e).__name__})", end="")

            if verbose:
                stop = e_usage.get("stop_reason", "?")
                retry = e_usage.get("retry_count", 0)
                recovered = e_usage.get("retry_recovered", 0)
                retry_s = f"  retry={recovered}/{retry}" if retry else ""
                yolo_s = (f"  yolo={matched_count}/{len(extracted)}"
                          if yolo_bboxes else "")
                print(f"detect={len(cards)}  extract={len(extracted)}{yolo_s}  "
                      f"stop={stop}{retry_s}  $${result.cost_usd:.3f}")
        except Exception as e:
            page_result = PageExtraction(page=page.page_no, error=str(e))
            result.pages.append(page_result)
            if verbose:
                print(f"FAILED: {type(e).__name__}: {e}")

        # Emit progress after each page (success OR failure) so the bot sees
        # monotonic progress even if a page errors out.
        pages_done = page.page_no
        pct = EXTRACT_START_PCT + (EXTRACT_END_PCT - EXTRACT_START_PCT) * (
            pages_done / total_pages
        )
        _emit(
            "extract", pct,
            f"Extracted page {pages_done}/{total_pages} · {cards_so_far} cards so far",
        )

    # Cold-vendor auto-enrich: if any extracted descriptions aren't in the
    # description_map cache yet, classify them now via one batched Claude call
    # per ~30 novel descriptions. Persists into vocab/description_map.json so
    # future runs of any catalog sharing those silhouettes are free.
    # The enrichment runs BEFORE the oracle so the verification sees the
    # enriched vocab. Skippable via auto_enrich=False for runs where you don't
    # want vocab to grow.
    if auto_enrich:
        try:
            from buysheet_v2.tools.enrich_description_map import enrich as _enrich_desc
            from buysheet_v2.tools.enrich_color_synonyms import enrich as _enrich_colors
            import buysheet_v2.verify as _vmod
            if verbose:
                print(f"[pipeline] auto-enriching description + color vocab for novel tokens...")
            _emit("enrich", 0.96, "Enriching vocab for any novel descriptions + colors")
            # Write current extraction to a temp sidecar so the enrich tools
            # can read it. Same path the bot would persist to — safe to write
            # over since both tools only read it.
            tmp_sidecar = pdf_path.with_suffix("").with_name(f"{pdf_path.stem}.v2.cards.json")
            tmp_sidecar.write_text(result.model_dump_json(indent=2, exclude_none=False))
            _enrich_desc([tmp_sidecar])
            _enrich_colors([tmp_sidecar])
            # Invalidate cached vocab so verify picks up the enriched maps
            _vmod._DESCRIPTION_MAP = None
            _vmod._COLOR_SYNONYMS = None
        except Exception as e:
            if verbose:
                print(f"[pipeline] vocab enrichment skipped: {type(e).__name__}: {e}")

    # Drop cards whose SKU clearly isn't a real vendor code. The VLM
    # sometimes fabricates slugified product names as SKUs on lookbook pages
    # where no real SKU is visible (Mizuno: "MXR-DENTELLE-PINK",
    # "WAVE PROPHECY LS OPEN MESH", "N/A"). A buyer can't track a product
    # without a real SKU, so these cards don't belong in the buy sheet.
    from buysheet_v2.consistency import (
        deterministic_fill, drop_invalid_sku_cards, normalize_extraction,
    )
    for pe in result.pages:
        survivors, dropped = drop_invalid_sku_cards(pe.cards)
        pe.cards = survivors
        if dropped and verbose:
            print(f"[pipeline] page {pe.page}: dropped {dropped} cards with invalid SKUs")

    # Apply card-level normalizations before scoring (e.g. demote VLM's
    # K-Footwear guesses to M-Footwear when there's no kids evidence in the
    # description or SKU). Mutates cards in place so the oracle sees the
    # corrected values and the xlsx writes the corrected defaults.
    fix_counts = normalize_extraction(result.all_cards)
    if verbose and fix_counts:
        print(f"[pipeline] card normalizations: {fix_counts}")

    # Backfill structured fields the VLM skipped on dense pages. The data is
    # provably in source for each missing SKU; we just re-apply the same
    # regex the oracle uses for verification but as an EXTRACTION fallback.
    # Only fires on None values — never overwrites a real VLM extraction.
    # Also fills None brand with the catalog-level brand and derives missing
    # standard_color from card.color via the vocab synonym map.
    from buysheet_v2.consistency import detect_catalog_brand
    page_text_by = {p.page: (p.page_text or "") for p in result.pages}
    fill_counts = deterministic_fill(
        result.all_cards, page_text_by,
        catalog_brand=detect_catalog_brand(result.all_cards),
    )
    if verbose and fill_counts:
        print(f"[pipeline] deterministic backfill: {fill_counts}")

    # Run the semantic oracle (no API cost — pure source-text verification)
    if verbose:
        print(f"[pipeline] running semantic oracle...")
    _emit("verify", 0.98, "Verifying every extracted value against source text")
    result = verify_catalog(result)
    summary = oracle_summary(result)
    passing = sum(s["correct"] for s in summary["per_field"].values())
    total = sum(s["total"] for s in summary["per_field"].values())
    contra = sum(s["contradicted"] for s in summary["per_field"].values())
    if verbose:
        print(f"  -> {passing}/{total} cells passing ({100 * passing / max(1, total):.1f}%)  "
              f"contradicted={contra} ({100 * contra / max(1, total):.1f}%)")

    # Attach matcher outcome markers to per-card flags. Three sources:
    #   1. page_match_markers — set when matcher raised ("matcher_failed")
    #      or returned an empty assignment ("no_match"). All cards on that
    #      page inherit the page-level marker.
    #   2. Per-card "no_photo_match" — YOLO ran on this page AND matcher
    #      assigned boxes to other SKUs, but THIS specific SKU got no box.
    #      Distinguishes the "matcher worked but missed me" case from
    #      "matcher broke for the whole page."
    # write.py uses these flags to surface "Photo match: <marker>" in the
    # photo cell comment so a buyer scanning the workbook can tell when a
    # crop came from a fallback (phototune / card_bbox) instead of YOLO.
    conf_by_key = {(c.sku, c.page): c for c in result.confidence}
    for card in result.all_cards:
        conf = conf_by_key.get((card.sku, card.page))
        if conf is None:
            continue
        page_marker = page_match_markers.get(card.page)
        if page_marker and page_marker not in conf.flags:
            conf.flags.append(page_marker)
        # Per-card no-match: YOLO ran on this page (entry exists), matcher
        # finished (no page-level marker), but this SKU got no bbox set.
        elif (
            card.page in yolo_detections
            and yolo_detections[card.page].get("bboxes")
            and not card.photo_bbox_px
            and "no_photo_match" not in conf.flags
        ):
            conf.flags.append("no_photo_match")

    # VLM-as-judge (TARGETED MODE): for cells the deterministic oracle could
    # neither confirm (≥0.7) nor contradict (≤0.05), Opus reads the page and
    # verifies just those (sku, field, value) suspects. Cheaper than the
    # legacy full re-extraction because the prompt is narrow and most cards
    # carry 0-1 unverifiable fields.
    #   - agreement → bump per_field 0.5 → 0.7, source = vlm_judge_confirmed
    #   - disagreement → leave per_field at 0.5, store opus_value so the cell
    #     comment shows both candidates side-by-side for the buyer
    # Skip entirely with run_judge=False; useful for cheap previews.
    if run_judge:
        from buysheet_v2.judge import (
            merge_field_verifications, verify_suspect_fields,
        )

        # Build (sku, field, value) suspects per page from confidence at 0.5
        card_by_key = {(c.sku, c.page): c for c in result.all_cards}
        suspects_by_page: dict[int, list[tuple[str, str, str]]] = {}
        for conf in result.confidence:
            card = card_by_key.get((conf.sku, conf.page))
            if card is None:
                continue
            for field, conf_v in conf.per_field.items():
                if conf_v != 0.5:
                    continue
                value = getattr(card, field, None)
                if value is None:
                    continue
                suspects_by_page.setdefault(conf.page, []).append(
                    (conf.sku, field, str(value))
                )

        n_suspects = sum(len(v) for v in suspects_by_page.values())
        if n_suspects == 0:
            if verbose:
                print("[pipeline] judge skipped — no fields at confidence 0.5")
        else:
            if verbose:
                print(f"[pipeline] judge: verifying {n_suspects} suspect "
                      f"field(s) across {len(suspects_by_page)} page(s)...")
            _emit("judge", 0.99,
                  f"Cross-checking {n_suspects} unverifiable cell(s) with Opus")
            verdicts_by_page: dict[int, list] = {}
            judge_cost = 0.0
            page_by_no = {p.page_no: p for p in doc.pages}
            for page_no, suspects in suspects_by_page.items():
                page = page_by_no.get(page_no)
                if page is None:
                    continue
                try:
                    verdicts, jusage = verify_suspect_fields(
                        page, suspects, client=client,
                    )
                    verdicts_by_page[page_no] = verdicts
                    judge_cost += jusage.get("cost_usd", 0.0)
                    result.cost_usd += jusage.get("cost_usd", 0.0)
                    result.tokens_input += jusage.get("input_tokens", 0)
                    result.tokens_output += jusage.get("output_tokens", 0)
                    result.tokens_cache_read += jusage.get("cache_read_tokens", 0)
                except Exception as e:
                    if verbose:
                        print(f"  [judge] page {page_no} failed: "
                              f"{type(e).__name__}: {e}")
            counts = merge_field_verifications(result.confidence, verdicts_by_page)
            if verbose:
                print(f"  -> judge cost ${judge_cost:.2f}  "
                      f"agreed_bump={counts['agreed_bump']}  "
                      f"disagreed={counts['disagreed']}  "
                      f"sku_missing={counts['sku_missing']}  "
                      f"out_of_band={counts['out_of_band']}")

    if cache_sidecar:
        sidecar.write_text(result.model_dump_json(indent=2, exclude_none=False))
        if verbose:
            print(f"[pipeline] cached extraction -> {sidecar.name}")
    if verbose:
        print(f"[pipeline] DONE  cards={len(result.all_cards)}  "
              f"tokens_in={result.tokens_input}  tokens_out={result.tokens_output}  "
              f"cost=${result.cost_usd:.3f}")
    pct_pass = 100 * passing / max(1, total)
    _emit(
        "done", 1.0,
        f"Extracted {len(result.all_cards)} cards · "
        f"{passing}/{total} cells verified ({pct_pass:.1f}%) · "
        f"{contra} contradicted · cost ${result.cost_usd:.2f}",
    )

    return result
