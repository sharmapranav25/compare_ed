"""Workbook generation: CatalogExtraction -> BUYSHEET_<vendor>.xlsx.

Opens BUYSHEET_template.xlsx, creates one tab per distinct brand from the
extracted cards (single brand -> rename TEMPLATE tab to vendor name), writes
8 columns + (optionally) photo per row.

v1 ships TEXT-ONLY: image binding is deferred to a future release. The photo
embedding helpers in lifted/photo_embed.py remain available and can be wired
back in once a reliable image-binding approach is in place (planned: native
PDF XObject extraction + nearest-neighbor SKU binding, or Meta SAM-based
"click above SKU" segmentation).

Cells with confidence <0.9 get amber background + cell comment showing the
raw value, page, and source.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import openpyxl
from openpyxl.comments import Comment
from openpyxl.styles import PatternFill

from buysheet_v2.consistency import normalize_consistency
from buysheet_v2.lifted.vocab_normalize import normalize_season, normalize_vendor
from buysheet_v2.schemas.card import ProductCard
from buysheet_v2.schemas.extraction_result import CardConfidence, CatalogExtraction

REPO = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = REPO / "BUYSHEET_template.xlsx"

# Column map: ProductCard field name -> 1-indexed Excel column
COL = {
    "photo":            1,   # A   (reserved; not populated in text-only v1)
    "sku":              2,   # B  STYLE #
    "mg":               3,   # C  MG
    "sg":               4,   # D  SG
    "ssg":              5,   # E  SSG
    "description":      6,   # F  Item Description
    "color":            7,   # G  Color Desc
    "standard_color":   8,   # H  Standard Color
    "intro_date":      16,   # P  INTRO DATE
    "usd_cost":        22,   # V  USD Cost
    "usd_retail":      23,   # W  USD Retail
}
DATA_ROW_START = 10  # template's first SKU row

AMBER_FILL = PatternFill(start_color="FFE5A8", end_color="FFE5A8", fill_type="solid")
LOW_CONFIDENCE_THRESHOLD = 0.9


def _confidence_lookup(confidence: list[CardConfidence]) -> dict[tuple[str, int], CardConfidence]:
    return {(c.sku, c.page): c for c in confidence}


def _safe_tab_name(name: str) -> str:
    """Excel worksheet names: max 31 chars, no /\\?*[]:"""
    cleaned = name
    for c in r'/\?*[]:':
        cleaned = cleaned.replace(c, "_")
    return cleaned[:31]


def _write_cell(ws, row: int, col: int, value, comment: Optional[str] = None,
                low_confidence: bool = False) -> None:
    cell = ws.cell(row, col)
    cell.value = value
    if low_confidence:
        cell.fill = AMBER_FILL
    if comment:
        cell.comment = Comment(comment, "Kith Buysheet Agent v2")


def write_workbook(
    extraction: CatalogExtraction,
    out_path: Path,
    *,
    catalog_brand_hint: Optional[str] = None,
) -> Path:
    """Write the extraction to a BUYSHEET-shaped xlsx (text-only v1).

    Multi-brand handling is delegated to consistency.normalize_consistency,
    which votes brand by SKU prefix family + partitions cards. When >=2 brands
    are present, one worksheet per brand is created; otherwise the single
    TEMPLATE sheet is used.

    catalog_brand_hint: fallback brand for cards with brand=None (used in
    single-brand catalogs where the brand isn't repeated on every card).

    NOTE: column A (PHOTO) is reserved but not populated in this release.
    See ARCHITECTURE.md "Known Limitations" for details.
    """
    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"template missing: {TEMPLATE_PATH}")

    wb = openpyxl.load_workbook(TEMPLATE_PATH)
    template_ws = wb["TEMPLATE"]
    conf_lookup = _confidence_lookup(extraction.confidence)

    all_cards = extraction.all_cards
    cons = normalize_consistency(all_cards, catalog_brand=catalog_brand_hint)
    partitions = cons["partitions"]
    multi_brand = cons["is_multi_brand"]
    distinct_brands = sorted(b for b in partitions if b and b != "_unbranded")

    season = normalize_season(extraction.vendor_key, wb)

    if multi_brand:
        ws_map: dict[str, openpyxl.worksheet.worksheet.Worksheet] = {}
        first = distinct_brands[0]
        template_ws.title = _safe_tab_name(first)
        ws_map[first] = template_ws
        for brand in distinct_brands[1:]:
            ws_map[brand] = wb.copy_worksheet(template_ws)
            ws_map[brand].title = _safe_tab_name(brand)
        unbranded = partitions.get("_unbranded", [])
        if unbranded:
            partitions[first] = partitions.get(first, []) + unbranded
        for brand_name, ws in ws_map.items():
            _populate_workbook(ws, partitions.get(brand_name, []), conf_lookup, brand_name)
            vendor_match = normalize_vendor(brand_name, wb) or brand_name
            ws["B1"] = vendor_match
            if season:
                ws["B2"] = season
    else:
        target_brand = distinct_brands[0] if distinct_brands else None
        _populate_workbook(template_ws, all_cards, conf_lookup, target_brand)
        if target_brand:
            vendor_match = normalize_vendor(target_brand, wb) or target_brand
            template_ws["B1"] = vendor_match
        if season:
            template_ws["B2"] = season

    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_path)
    return out_path


def _populate_workbook(
    ws,
    cards: list[ProductCard],
    conf_lookup: dict[tuple[str, int], CardConfidence],
    brand_name: Optional[str],
) -> None:
    """Write a card list to one worksheet starting at DATA_ROW_START.

    Text-only: column A (photo) is reserved/empty. All extracted text fields
    are written with confidence-based amber highlighting + provenance comments.
    """
    for i, card in enumerate(cards):
        row = DATA_ROW_START + i
        if row > 883:
            break  # template hard cap
        conf = conf_lookup.get((card.sku, card.page))

        for field_name, col in COL.items():
            if field_name == "photo":
                continue  # reserved for future image-binding work
            if field_name == "sku":
                value = card.sku
                low = False
            else:
                value = getattr(card, field_name, None)
                if value is None:
                    continue
                conf_value = (conf.per_field.get(field_name) if conf else None) or 0.0
                low = conf_value < LOW_CONFIDENCE_THRESHOLD
            comment = None
            if conf:
                src = conf.per_field_source.get(field_name, "vlm")
                conf_val = conf.per_field.get(field_name, 0.0)
                comment = (
                    f"page={card.page}  source={src}  confidence={conf_val:.2f}"
                )
            _write_cell(ws, row, col, value, comment=comment, low_confidence=low)
