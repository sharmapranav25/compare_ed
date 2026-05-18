"""Vocabulary normalization helpers — lifted from phase5/populate.py.

These map raw vendor strings to template-compatible values:
- normalize_intro_date: any date format -> JAN/FEB/.../DEC code
- normalize_vendor: vendor slug -> Vendor Data sheet value (B1 dropdown)
- normalize_season: doc-name token -> Season sheet value (B2 dropdown)
- normalize_gender: vendor gender variant -> M/W/K-Footwear

These functions are pure; they read no global state. The vendor and season
versions take an openpyxl workbook (the template) to consult the validation
lists, so they remain in sync with whatever the template declares.
"""
from __future__ import annotations

from datetime import date, datetime

MONTH_CODES = (
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
    "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
)
MONTH_NAMES = {
    "JANUARY": "JAN", "JAN": "JAN",
    "FEBRUARY": "FEB", "FEB": "FEB",
    "MARCH": "MAR", "MAR": "MAR",
    "APRIL": "APR", "APR": "APR",
    "MAY": "MAY",
    "JUNE": "JUN", "JUN": "JUN",
    "JULY": "JUL", "JUL": "JUL",
    "AUGUST": "AUG", "AUG": "AUG",
    "SEPTEMBER": "SEP", "SEPT": "SEP", "SEP": "SEP",
    "OCTOBER": "OCT", "OCT": "OCT",
    "NOVEMBER": "NOV", "NOV": "NOV",
    "DECEMBER": "DEC", "DEC": "DEC",
}


def normalize_gender(raw) -> str | None:
    """Map vendor gender variants to {M-Footwear, W-Footwear, K-Footwear}."""
    if raw in (None, ""):
        return None
    s = str(raw).strip().upper()
    if s in ("M-FOOTWEAR", "M", "MEN", "MENS", "MEN'S", "MALE"):
        return "M-Footwear"
    if s in ("W-FOOTWEAR", "W", "F", "WMNS", "WOMEN", "WOMENS", "WOMEN'S", "FEMALE"):
        return "W-Footwear"
    if s in (
        "K-FOOTWEAR", "K", "KIDS", "YOUTH", "CHILD", "CHILDREN",
        "INFANT", "TODDLER", "GS", "TD", "PS",
    ):
        return "K-Footwear"
    if s == "U" or "UNISEX" in s:
        return "K-Footwear"
    return None


def normalize_intro_date(raw) -> str | None:
    """Return the template's JAN-DEC code for any common date format.

    Handles datetime/date objects, ISO strings (2026-06-01), US slash dates
    (5/1/25), and month-name strings (June, JUL).
    """
    if raw in (None, ""):
        return None
    if isinstance(raw, datetime):
        return MONTH_CODES[raw.month - 1]
    if isinstance(raw, date):
        return MONTH_CODES[raw.month - 1]

    s = str(raw).strip()
    if not s:
        return None
    upper = s.upper()
    if upper in MONTH_CODES:
        return upper
    if upper in MONTH_NAMES:
        return MONTH_NAMES[upper]

    parts = upper.replace("/", "-").split("-")
    if (
        len(parts) >= 2
        and len(parts[0]) == 4
        and parts[0].isdigit()
        and parts[1].isdigit()
    ):
        month = int(parts[1])
        if 1 <= month <= 12:
            return MONTH_CODES[month - 1]

    slash_parts = upper.split("/")
    if len(slash_parts) >= 2 and slash_parts[0].isdigit():
        month = int(slash_parts[0])
        if 1 <= month <= 12:
            return MONTH_CODES[month - 1]

    for token in upper.replace(",", " ").split():
        if token in MONTH_NAMES:
            return MONTH_NAMES[token]
    return None


def _column_values(wb, sheet_name: str, col: int, start_row: int = 2) -> list:
    """Read non-empty values from a column of a reference sheet."""
    ws = wb[sheet_name]
    out = []
    for row in range(start_row, ws.max_row + 1):
        value = ws.cell(row, col).value
        if value not in (None, ""):
            out.append(value)
    return out


def normalize_vendor(vendor: str, wb) -> str | None:
    """Map a vendor slug (e.g. 'adidas_premium', 'nike_ho26') to a Vendor Data value."""
    allowed = _column_values(wb, "Vendor Data", 1)
    base = vendor.split("_", 1)[0].rstrip("0123456789")
    candidates = [
        vendor,
        vendor.replace("_", " "),
        vendor.split("_", 1)[0],
        base,
        vendor.rstrip("0123456789_"),
    ]
    for candidate in candidates:
        c = candidate.strip()
        if not c:
            continue
        for allowed_value in allowed:
            if str(allowed_value).lower() == c.lower():
                return allowed_value
    return None


def normalize_season(doc_name: str, wb) -> str | None:
    """Extract a season code from a doc name (e.g. 'HO26' -> '2026-Q4')."""
    allowed = {str(v) for v in _column_values(wb, "Season", 1)}
    tokens = doc_name.replace("-", " ").replace("_", " ").split()
    for token in tokens:
        upper = token.upper()
        if upper in allowed:
            return upper
        if len(upper) == 4 and upper[:2].isalpha() and upper[2:].isdigit():
            year = f"20{upper[2:]}"
            prefix = upper[:2]
            if prefix in {"FW", "FA"}:
                candidates = [f"Fall-{year}", f"{year}-Q3"]
            elif prefix in {"SP", "SS"}:
                candidates = [f"Spring-{year}", f"{year}-Q2"]
            elif prefix in {"HO"}:
                candidates = [f"{year}-Q4", f"Fall-{year}"]
            elif prefix in {"SU"}:
                candidates = [f"{year}-Q3", f"Spring-{year}"]
            else:
                candidates = []
            for candidate in candidates:
                if candidate in allowed:
                    return candidate
    return None
