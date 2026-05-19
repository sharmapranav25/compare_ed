"""Direct Google Sheets API writer — replaces the xlsx → manual-import → Apps
Script chain for production buyer delivery.

The bot creates a fresh Sheet by copying a master template, writes all extracted
cells via spreadsheets.batchUpdate, uploads each product photo to a per-sheet
Drive subfolder, and writes `=IMAGE(drive_url)` formulas so photos render
in-cell from the moment the Sheet exists. No drift possible, no manual menu
click required.

Required env:
  GOOGLE_SERVICE_ACCOUNT_JSON  path to the service-account JSON key file
  KITH_TEMPLATE_SHEET_ID       Drive file ID of the master BUYSHEET template
                               (a Google Sheet, NOT xlsx; manually converted)
  KITH_SHEETS_DOMAIN           Workspace domain for sharing (e.g. "keelo-ai.com")
                               OR "_public" to share by-link with anyone
  KITH_SHEETS_PARENT_FOLDER    (optional) Drive folder ID where new Sheets land;
                               defaults to service account's root
"""
from __future__ import annotations

import io
import logging
import os
import time
from pathlib import Path
from typing import Optional

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseUpload

from buysheet_v2.schemas.card import ProductCard
from buysheet_v2.schemas.extraction_result import CardConfidence, CatalogExtraction
from buysheet_v2.consistency import normalize_consistency

log = logging.getLogger(__name__)

# Match the xlsx writer's column layout so the template (same source structure)
# binds cells correctly. Column letters → 1-indexed col numbers.
COL = {
    "photo":          1,   # A
    "sku":            2,   # B
    "mg":             3,   # C
    "sg":             4,   # D
    "ssg":            5,   # E
    "description":    6,   # F
    "color":          7,   # G
    "standard_color": 8,   # H
    "intro_date":    16,   # P
    "usd_cost":      22,   # V
    "usd_retail":    23,   # W
}

DATA_ROW_START = 10  # template first SKU row (matches xlsx template)
MAX_ROW = 883        # template hard cap (matches xlsx template)

LOW_CONFIDENCE_THRESHOLD = 0.9
CONTRADICTED_THRESHOLD = 0.05

# Cell background colors for tier formatting. RGB 0-1 as Sheets API expects.
AMBER_BG = {"red": 1.0, "green": 0.87, "blue": 0.5}
RED_BG   = {"red": 1.0, "green": 0.78, "blue": 0.78}

_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


# --------------------------------------------------------------------------- auth

def _credentials():
    """Load credentials with three paths in priority order:

    1. Service-account JSON (env: GOOGLE_SERVICE_ACCOUNT_JSON) — best for
       production / multi-user deployments. Blocked by some orgs.
    2. OAuth installed-app client (env: GOOGLE_OAUTH_CLIENT_JSON) — works
       when service-account keys are disabled by org policy. First run opens
       a browser for one-time consent; subsequent runs use a cached refresh
       token at GOOGLE_OAUTH_TOKEN_CACHE (defaults to
       ~/.config/gcp/kith-buysheet-token.json).
    3. Application Default Credentials (gcloud auth application-default login)
       — simplest local-dev path. Bot acts as the human who ran gcloud auth.
       Will fail if the gcloud client ID is blocked by org OAuth restrictions.
    """
    sa_path = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if sa_path:
        if not Path(sa_path).exists():
            raise FileNotFoundError(f"Service account JSON not found: {sa_path}")
        return service_account.Credentials.from_service_account_file(
            sa_path, scopes=_SCOPES,
        )

    oauth_path = os.environ.get("GOOGLE_OAUTH_CLIENT_JSON")
    if oauth_path:
        if not Path(oauth_path).exists():
            raise FileNotFoundError(
                f"OAuth client JSON not found: {oauth_path}"
            )
        from google.oauth2.credentials import Credentials as UserCredentials
        from google.auth.transport.requests import Request
        from google_auth_oauthlib.flow import InstalledAppFlow
        token_cache = Path(
            os.environ.get(
                "GOOGLE_OAUTH_TOKEN_CACHE",
                str(Path.home() / ".config" / "gcp" / "kith-buysheet-token.json"),
            )
        ).expanduser()
        creds: Optional[UserCredentials] = None
        if token_cache.exists():
            try:
                creds = UserCredentials.from_authorized_user_file(
                    str(token_cache), _SCOPES,
                )
            except Exception as e:
                log.warning("Failed to load cached OAuth token (%s); re-auth", e)
                creds = None
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                log.warning("Token refresh failed (%s); re-running consent flow", e)
                creds = None
        if not creds or not creds.valid:
            flow = InstalledAppFlow.from_client_secrets_file(oauth_path, _SCOPES)
            creds = flow.run_local_server(port=0)
            token_cache.parent.mkdir(parents=True, exist_ok=True)
            token_cache.write_text(creds.to_json())
            log.info("Cached OAuth token to %s", token_cache)
        return creds

    # ADC fallback — picked up from ~/.config/gcloud/application_default_credentials.json
    from google.auth import default as _adc_default
    creds, _ = _adc_default(scopes=_SCOPES)
    return creds


def _sheets_service():
    return build("sheets", "v4", credentials=_credentials(), cache_discovery=False)


def _drive_service():
    return build("drive", "v3", credentials=_credentials(), cache_discovery=False)


# --------------------------------------------------------------------------- core

def copy_template(*, new_name: str, parent_folder_id: Optional[str] = None) -> str:
    """Copy the master template Sheet into a new file. Returns the new Sheet ID.

    Master template ID comes from KITH_TEMPLATE_SHEET_ID env var. The template
    must be a native Google Sheet (not xlsx) shared with the service account
    as Editor.
    """
    template_id = os.environ.get("KITH_TEMPLATE_SHEET_ID")
    if not template_id:
        raise RuntimeError(
            "KITH_TEMPLATE_SHEET_ID env var required (Drive ID of master "
            "BUYSHEET template Sheet)."
        )
    drive = _drive_service()
    body = {"name": new_name}
    if parent_folder_id:
        body["parents"] = [parent_folder_id]
    new_file = drive.files().copy(
        fileId=template_id, body=body, supportsAllDrives=True,
    ).execute()
    log.info("Copied template → new Sheet id=%s name=%s", new_file["id"], new_name)
    return new_file["id"]


def _sheet_tabs(spreadsheet_id: str) -> dict[str, int]:
    """Return {tab_title: tab_internal_id} for the spreadsheet."""
    sheets = _sheets_service()
    meta = sheets.spreadsheets().get(
        spreadsheetId=spreadsheet_id, fields="sheets.properties",
    ).execute()
    return {s["properties"]["title"]: s["properties"]["sheetId"]
            for s in meta.get("sheets", [])}


def _rename_tab(spreadsheet_id: str, tab_id: int, new_title: str) -> None:
    sheets = _sheets_service()
    sheets.spreadsheets().batchUpdate(
        spreadsheetId=spreadsheet_id,
        body={"requests": [{
            "updateSheetProperties": {
                "properties": {"sheetId": tab_id, "title": new_title},
                "fields": "title",
            }
        }]},
    ).execute()


def _duplicate_tab(spreadsheet_id: str, source_tab_id: int,
                   new_title: str) -> int:
    """Duplicate an existing tab and rename. Returns the new tab's internal ID."""
    sheets = _sheets_service()
    resp = sheets.spreadsheets().batchUpdate(
        spreadsheetId=spreadsheet_id,
        body={"requests": [{
            "duplicateSheet": {
                "sourceSheetId": source_tab_id,
                "newSheetName": new_title,
            }
        }]},
    ).execute()
    return resp["replies"][0]["duplicateSheet"]["properties"]["sheetId"]


# --------------------------------------------------------------------------- cell write

def _cell_a1(col_1indexed: int, row_1indexed: int) -> str:
    """Convert (col, row) → A1 notation (e.g. (2, 10) → 'B10')."""
    s = ""
    n = col_1indexed
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return f"{s}{row_1indexed}"


def _build_cell_value_requests(
    tab_title: str,
    cards: list[ProductCard],
    conf_lookup: dict[tuple[str, int], CardConfidence],
) -> list[dict]:
    """Return a list of valueRange dicts ready for values.batchUpdate.

    Each card → one row. Cell values respect the same red/amber blanking the
    xlsx writer does (verify-derived: red blanks the cell, amber keeps value).
    """
    data: list[dict] = []
    truncated = 0
    for i, card in enumerate(cards):
        row = DATA_ROW_START + i
        if row > MAX_ROW:
            truncated = len(cards) - i
            break
        conf = conf_lookup.get((card.sku, card.page))
        for field_name, col in COL.items():
            if field_name == "photo":
                continue
            if field_name == "sku":
                value = card.sku
                conf_value = (conf.per_field.get("sku") if conf else None)
                if conf_value is None:
                    conf_value = 1.0
            else:
                value = getattr(card, field_name, None)
                if value is None:
                    continue
                conf_value = (conf.per_field.get(field_name) if conf else None) or 0.0
            # Red tier → blank the cell (don't ship a value the oracle rejected)
            if conf_value <= CONTRADICTED_THRESHOLD:
                continue
            data.append({
                "range": f"'{tab_title}'!{_cell_a1(col, row)}",
                "values": [[value]],
            })
    return data, truncated


def _build_formatting_requests(
    tab_id: int,
    cards: list[ProductCard],
    conf_lookup: dict[tuple[str, int], CardConfidence],
) -> list[dict]:
    """Build repeatCell requests for amber/red backgrounds based on per-field
    confidence. Mirrors the xlsx writer's tier logic so the Sheet and the
    xlsx backup look identical.
    """
    requests: list[dict] = []
    for i, card in enumerate(cards):
        row_idx = DATA_ROW_START - 1 + i  # 0-indexed for Sheets API
        if row_idx + 1 > MAX_ROW:
            break
        conf = conf_lookup.get((card.sku, card.page))
        if not conf:
            continue
        for field_name, col in COL.items():
            if field_name == "photo":
                continue
            conf_value = conf.per_field.get(field_name)
            if conf_value is None:
                continue
            if conf_value <= CONTRADICTED_THRESHOLD:
                bg = RED_BG
            elif conf_value < LOW_CONFIDENCE_THRESHOLD:
                bg = AMBER_BG
            else:
                continue
            requests.append({
                "repeatCell": {
                    "range": {
                        "sheetId": tab_id,
                        "startRowIndex": row_idx,
                        "endRowIndex": row_idx + 1,
                        "startColumnIndex": col - 1,
                        "endColumnIndex": col,
                    },
                    "cell": {
                        "userEnteredFormat": {"backgroundColor": bg}
                    },
                    "fields": "userEnteredFormat.backgroundColor",
                }
            })
    return requests


def _build_note_requests(
    tab_id: int,
    cards: list[ProductCard],
    conf_lookup: dict[tuple[str, int], CardConfidence],
) -> list[dict]:
    """Surface the same per-cell comments the xlsx writer adds, as Sheets-API
    cell notes (visible as a yellow corner triangle on hover).
    """
    from buysheet_v2.write import _comment_for  # reuse the canonical formatter
    requests: list[dict] = []
    for i, card in enumerate(cards):
        row_idx = DATA_ROW_START - 1 + i
        if row_idx + 1 > MAX_ROW:
            break
        conf = conf_lookup.get((card.sku, card.page))
        if not conf:
            continue
        for field_name, col in COL.items():
            if field_name == "photo":
                continue
            conf_value = conf.per_field.get(field_name)
            if conf_value is None:
                continue
            source = conf.per_field_source.get(field_name, "vlm")
            if field_name == "sku":
                value = card.sku
            else:
                value = getattr(card, field_name, None)
            note = _comment_for(
                field_name, value, conf_value, source,
                page=card.page, blanked=(conf_value <= CONTRADICTED_THRESHOLD),
                sku_context=conf.sku_context if field_name == "sku" else None,
            )
            requests.append({
                "updateCells": {
                    "range": {
                        "sheetId": tab_id,
                        "startRowIndex": row_idx,
                        "endRowIndex": row_idx + 1,
                        "startColumnIndex": col - 1,
                        "endColumnIndex": col,
                    },
                    "rows": [{"values": [{"note": note}]}],
                    "fields": "note",
                }
            })
    return requests


# --------------------------------------------------------------------------- images

def _upload_png_to_drive(
    png_bytes: bytes, file_name: str, parent_folder_id: str,
) -> str:
    """Upload a PNG to Drive, return the file ID. Caller is responsible for
    ensuring the parent folder is domain-shared so the resulting IMAGE() URL
    is viewable by buyers.
    """
    drive = _drive_service()
    media = MediaIoBaseUpload(
        io.BytesIO(png_bytes), mimetype="image/png", resumable=False,
    )
    body = {"name": file_name, "parents": [parent_folder_id]}
    f = drive.files().create(
        body=body, media_body=media, fields="id",
        supportsAllDrives=True,
    ).execute()
    return f["id"]


def _create_images_subfolder(
    spreadsheet_id: str, parent_folder_id: Optional[str] = None,
) -> str:
    """Create a Drive subfolder named after the Sheet to hold its photo PNGs."""
    drive = _drive_service()
    body = {
        "name": f"_photos_for_{spreadsheet_id[:10]}",
        "mimeType": "application/vnd.google-apps.folder",
    }
    if parent_folder_id:
        body["parents"] = [parent_folder_id]
    f = drive.files().create(body=body, fields="id", supportsAllDrives=True).execute()
    return f["id"]


# --------------------------------------------------------------------------- sharing

def share_sheet(spreadsheet_id: str, *, domain: Optional[str] = None,
                emails: Optional[list[str]] = None) -> None:
    """Share the Sheet for editing. Pass `domain` for Workspace domain-wide
    access (most common), OR `emails` for specific viewers. If neither is set,
    falls back to KITH_SHEETS_DOMAIN env var, then to "_public" (anyone-with-link).
    """
    drive = _drive_service()
    domain = domain or os.environ.get("KITH_SHEETS_DOMAIN")
    if emails:
        for email in emails:
            drive.permissions().create(
                fileId=spreadsheet_id,
                body={"type": "user", "role": "writer", "emailAddress": email},
                sendNotificationEmail=False,
                supportsAllDrives=True,
            ).execute()
        return
    if domain and domain != "_public":
        drive.permissions().create(
            fileId=spreadsheet_id,
            body={"type": "domain", "role": "writer", "domain": domain},
            supportsAllDrives=True,
        ).execute()
        return
    # Fallback: anyone-with-link
    drive.permissions().create(
        fileId=spreadsheet_id,
        body={"type": "anyone", "role": "writer"},
        supportsAllDrives=True,
    ).execute()


def share_folder_publicly_within_domain(
    folder_id: str, *, domain: Optional[str] = None,
) -> None:
    """Apply same sharing policy to a Drive folder. Needed so the IMAGE()
    formula URLs in the Sheet are viewable by everyone who can see the Sheet.
    """
    share_sheet(folder_id, domain=domain)  # same API surface, file-or-folder agnostic


# --------------------------------------------------------------------------- orchestrator

def write_extraction_to_sheet(
    extraction: CatalogExtraction,
    *,
    pdf_path: Optional[Path] = None,
    sheet_name: Optional[str] = None,
    embed_photos: bool = True,
) -> dict:
    """End-to-end: copy template → write cells → format → embed images → share.

    Returns: {"sheet_id", "sheet_url", "embedded_photos", "truncated_cards"}.
    """
    parent_folder = os.environ.get("KITH_SHEETS_PARENT_FOLDER")
    sheet_name = sheet_name or f"BUYSHEET_{extraction.vendor_key}_v2"

    spreadsheet_id = copy_template(
        new_name=sheet_name, parent_folder_id=parent_folder,
    )

    cards = extraction.all_cards
    cons = normalize_consistency(cards)
    partitions = cons["partitions"]
    distinct_brands = sorted(b for b in partitions if b and b != "_unbranded")
    multi_brand = cons["is_multi_brand"]

    tabs = _sheet_tabs(spreadsheet_id)
    if "TEMPLATE" not in tabs:
        raise RuntimeError(
            f"Template Sheet (id={os.environ.get('KITH_TEMPLATE_SHEET_ID')}) "
            f"has no 'TEMPLATE' tab — got {list(tabs.keys())}"
        )
    template_tab_id = tabs["TEMPLATE"]

    conf_lookup = {(c.sku, c.page): c for c in extraction.confidence}

    # Build per-tab card partitions: same logic as write.py's write_workbook
    if multi_brand:
        first = distinct_brands[0]
        _rename_tab(spreadsheet_id, template_tab_id, first)
        tab_id_by_brand = {first: template_tab_id}
        for brand in distinct_brands[1:]:
            tab_id_by_brand[brand] = _duplicate_tab(
                spreadsheet_id, template_tab_id, brand,
            )
        unbranded = partitions.get("_unbranded", [])
        if unbranded:
            partitions[first] = partitions.get(first, []) + unbranded
        tabs_to_write = [(b, partitions.get(b, []), tab_id_by_brand[b])
                         for b in distinct_brands]
    else:
        tabs_to_write = [("TEMPLATE", cards, template_tab_id)]

    truncated_total = 0
    all_value_data: list[dict] = []
    all_format_requests: list[dict] = []
    all_note_requests: list[dict] = []
    for tab_title, tab_cards, tab_id in tabs_to_write:
        value_data, truncated = _build_cell_value_requests(
            tab_title, tab_cards, conf_lookup,
        )
        all_value_data.extend(value_data)
        all_format_requests.extend(
            _build_formatting_requests(tab_id, tab_cards, conf_lookup)
        )
        all_note_requests.extend(
            _build_note_requests(tab_id, tab_cards, conf_lookup)
        )
        truncated_total += truncated

    sheets = _sheets_service()

    # Cell values: batch via values.batchUpdate (single round trip for hundreds of cells)
    if all_value_data:
        sheets.spreadsheets().values().batchUpdate(
            spreadsheetId=spreadsheet_id,
            body={"valueInputOption": "RAW", "data": all_value_data},
        ).execute()
        log.info("Wrote %d cell ranges to Sheet %s",
                 len(all_value_data), spreadsheet_id)

    # Formatting + notes: spreadsheet.batchUpdate (different endpoint than values)
    # Chunk to avoid the 50K-requests-per-call limit (cells.batchUpdate can be big).
    def _flush_batch(requests: list[dict], label: str) -> None:
        if not requests:
            return
        for chunk_start in range(0, len(requests), 500):
            chunk = requests[chunk_start:chunk_start + 500]
            sheets.spreadsheets().batchUpdate(
                spreadsheetId=spreadsheet_id, body={"requests": chunk},
            ).execute()
        log.info("Applied %d %s requests to Sheet %s",
                 len(requests), label, spreadsheet_id)

    _flush_batch(all_format_requests, "tier-formatting")
    _flush_batch(all_note_requests, "cell-notes")

    # Images: separate pass (Drive uploads + IMAGE() formula writes)
    embedded = 0
    if embed_photos and pdf_path is not None:
        embedded = _embed_photos_via_drive(
            spreadsheet_id, extraction, pdf_path, parent_folder, tabs_to_write,
        )

    # Share the Sheet (and the photos folder, if we created one)
    share_sheet(spreadsheet_id)

    sheet_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/edit"
    return {
        "sheet_id": spreadsheet_id,
        "sheet_url": sheet_url,
        "embedded_photos": embedded,
        "truncated_cards": truncated_total,
    }


def _embed_photos_via_drive(
    spreadsheet_id: str,
    extraction: CatalogExtraction,
    pdf_path: Path,
    parent_folder_id: Optional[str],
    tabs_to_write: list,
) -> int:
    """Crop each card's photo from the source PDF, upload to a per-Sheet Drive
    subfolder, and write =IMAGE(url) formulas into col A. Returns count embedded.

    This is the BIG win over xlsx: images go in as cell formulas at sheet
    creation time, so there's no floating-image-drift on import. The subfolder
    is shared with the same audience as the Sheet so the URLs render for any
    viewer.
    """
    import pypdfium2 as pdfium
    from buysheet_v2.write import _crop_card_png  # reuse the canonical cropper

    # Reuse phototune to resolve photo bboxes the same way the xlsx writer does
    from buysheet_v2.phototune import resolve_page_photo_bboxes

    sheets = _sheets_service()
    subfolder_id = _create_images_subfolder(spreadsheet_id, parent_folder_id)
    share_folder_publicly_within_domain(subfolder_id)

    layout_type = (extraction.layout.layout_type if extraction.layout else "unknown")

    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        from buysheet_v2.lifted.pdf_render import VLM_MAX_LONG_EDGE_PX
        page_dims: dict[int, tuple[int, int]] = {}
        for page_no in {c.page for c in extraction.all_cards}:
            page = pdf[page_no - 1]
            long_pt = max(page.get_width(), page.get_height())
            scale = VLM_MAX_LONG_EDGE_PX / long_pt
            w = int(round(page.get_width() * scale))
            h = int(round(page.get_height() * scale))
            page_dims[page_no] = (w, h)

        # Two-stage photo-bbox resolution — MATCHES xlsx writer (write.py).
        # Stage 1: photo_vlm primary (per-card VLM, tight + correctly-anchored
        #          bboxes; sidecar-cached so cost is ~$0 on re-runs).
        # Stage 2: phototune fallback for SKUs photo_vlm couldn't resolve
        #          (image-only pages, transient API failures after retry).
        # Without stage 1, photos are drawn from phototune-only bboxes which
        # often drift into neighboring cards (the Hoka misalignment bug).
        resolved: dict[tuple[int, str], tuple[int, int, int, int]] = {}
        try:
            from buysheet_v2.photo_vlm import resolve_catalog_photo_bboxes
            vlm_resolved, vlm_usage = resolve_catalog_photo_bboxes(
                pdf_path, extraction, page_dims,
            )
            resolved.update(vlm_resolved)
            log.info("photo_vlm resolved %d bboxes (%d fresh VLM calls)",
                     len(vlm_resolved), vlm_usage["calls"])
        except Exception as e:
            log.warning("photo_vlm failed catalog-wide (%s: %s); "
                        "phototune fallback only", type(e).__name__, e)
        # Stage 2: phototune for any SKU stage 1 didn't resolve. setdefault
        # so we never overwrite a VLM-resolved bbox with a looser heuristic one.
        for pe in extraction.pages:
            dims = page_dims.get(pe.page)
            if not dims:
                continue
            page_w, page_h = dims
            try:
                per_page = resolve_page_photo_bboxes(
                    pdf_path, pe.page, pe.cards, layout_type, page_w, page_h,
                )
                for sku, bbox in per_page.items():
                    resolved.setdefault((pe.page, sku), bbox)
            except Exception as e:
                log.warning("phototune failed on page %d: %s", pe.page, e)
                continue

        embedded = 0
        image_value_data: list[dict] = []
        for tab_title, tab_cards, _ in tabs_to_write:
            for i, card in enumerate(tab_cards):
                row = DATA_ROW_START + i
                if row > MAX_ROW:
                    break
                bbox = resolved.get((card.page, card.sku)) or card.photo_bbox_px or card.card_bbox_px
                if not bbox or not page_dims.get(card.page):
                    continue
                w, h = page_dims[card.page]
                img = _crop_card_png(pdf, card.page, list(bbox), w, h)
                if img is None:
                    continue
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                png_bytes = buf.getvalue()
                try:
                    file_id = _upload_png_to_drive(
                        png_bytes,
                        file_name=f"p{card.page}_{card.sku}.png",
                        parent_folder_id=subfolder_id,
                    )
                except HttpError as e:
                    log.warning("Drive upload failed for %s: %s", card.sku, e)
                    continue
                # Use the thumbnail URL form so Sheets renders the inline image
                # without requiring the viewer to be authed to Drive directly.
                image_url = f"https://drive.google.com/thumbnail?id={file_id}&sz=w200"
                image_value_data.append({
                    "range": f"'{tab_title}'!{_cell_a1(COL['photo'], row)}",
                    "values": [[f'=IMAGE("{image_url}")']],
                })
                embedded += 1

        if image_value_data:
            # Write the IMAGE formulas in batches of 500 to stay under quota
            for chunk_start in range(0, len(image_value_data), 500):
                chunk = image_value_data[chunk_start:chunk_start + 500]
                sheets.spreadsheets().values().batchUpdate(
                    spreadsheetId=spreadsheet_id,
                    body={"valueInputOption": "USER_ENTERED", "data": chunk},
                ).execute()
                time.sleep(0.5)  # polite throttle, Sheets API quota is 60 req/min default

        log.info("Embedded %d photos into Sheet %s", embedded, spreadsheet_id)
        return embedded
    finally:
        pdf.close()
