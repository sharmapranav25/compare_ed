"""Interactive ground-truth verification tool.

Walks an unverified tests/golden/<vendor>.json file SKU by SKU. For each:
  - Displays the extracted field values
  - Shows the source page number
  - Prompts: accept all, edit specific fields, skip, or quit
  - Saves progressively (Ctrl-C is safe — anything verified so far persists)

Usage:
    python -m buysheet_v2.tools.verify_golden tests/golden/nike_ho26.json
    python -m buysheet_v2.tools.verify_golden tests/golden/nike_ho26.json --skip-verified

Workflow:
    1. Open the source PDF for the catalog (path shown at top of session)
    2. For each SKU shown, navigate to the cited page in the PDF
    3. Eyeball the actual values vs the model's extraction:
         - Press Enter (or 'a'/'accept') if everything looks right
         - Press 'e'/'edit' to correct one or more fields
         - Press 's'/'skip' to leave unverified for later
         - Press 'q'/'quit' to save and exit
    4. Verified entries get _verified: true; eval harness only counts those.

Time budget: ~30 seconds per SKU on a clean grid catalog, ~1-2 min on a
dense lookbook. Plan for ~30 minutes per vendor at 25 SKUs.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

FIELDS_TO_VERIFY = [
    "sku", "brand", "description", "color", "standard_color",
    "mg", "sg", "ssg", "intro_date", "usd_cost", "usd_retail",
]


def _print_card(entry: dict, idx: int, total: int) -> None:
    print()
    print("=" * 80)
    print(f"SKU {idx + 1}/{total} — page {entry.get('_page', '?')}")
    print("=" * 80)
    for f in FIELDS_TO_VERIFY:
        val = entry.get(f)
        marker = "  " if val not in (None, "") else " *"  # mark blanks
        print(f"  {marker}{f:<18}  {val!r}")
    print()
    if entry.get("_verified"):
        print(f"  [STATUS] previously verified ({entry.get('_verified_at', '')})")
    else:
        print(f"  [STATUS] unverified")


def _prompt_edit(entry: dict) -> bool:
    """Prompt the user to edit specific fields. Returns True if any change made."""
    print("\nEnter field name to edit (or blank to finish):")
    changed = False
    while True:
        try:
            field = input("  field> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return changed
        if not field:
            return changed
        if field not in FIELDS_TO_VERIFY:
            print(f"    unknown field. Valid: {', '.join(FIELDS_TO_VERIFY)}")
            continue
        current = entry.get(field)
        print(f"    current value: {current!r}")
        try:
            new_raw = input("    new value (or 'null' for None, or blank to skip): ").strip()
        except (EOFError, KeyboardInterrupt):
            return changed
        if not new_raw:
            continue
        if new_raw.lower() in ("null", "none"):
            entry[field] = None
        elif field in ("usd_cost", "usd_retail"):
            try:
                entry[field] = float(new_raw)
            except ValueError:
                print(f"    not a number: {new_raw!r}")
                continue
        else:
            entry[field] = new_raw
        changed = True
        print(f"    updated.")


def verify_one(entry: dict, idx: int, total: int) -> str:
    """Show one entry; return action: 'accepted', 'edited', 'skipped', 'quit'."""
    _print_card(entry, idx, total)
    print()
    print("Action: [a]ccept all, [e]dit, [s]kip, [q]uit ")
    try:
        choice = input("  > ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return "quit"

    if choice in ("", "a", "accept"):
        return "accepted"
    if choice in ("e", "edit"):
        any_change = _prompt_edit(entry)
        return "edited" if any_change else "accepted"
    if choice in ("s", "skip"):
        return "skipped"
    if choice in ("q", "quit"):
        return "quit"
    print(f"  unrecognized choice {choice!r} — treating as skip")
    return "skipped"


def save_progress(golden_path: Path, golden: dict) -> None:
    golden_path.write_text(json.dumps(golden, indent=2, default=str))


def main() -> int:
    ap = argparse.ArgumentParser(prog="verify_golden")
    ap.add_argument("golden", type=Path, help="Path to tests/golden/<vendor>.json")
    ap.add_argument("--skip-verified", action="store_true",
                    help="Don't re-show entries that are already _verified=true")
    ap.add_argument("--verifier", default=None,
                    help="Name to stamp into _verified_by (defaults to $USER)")
    args = ap.parse_args()

    if not args.golden.exists():
        print(f"golden file not found: {args.golden}", file=sys.stderr)
        return 1

    golden = json.loads(args.golden.read_text())
    skus = golden.get("skus", [])
    if not skus:
        print(f"no SKUs in {args.golden}", file=sys.stderr)
        return 1

    import os
    verifier = args.verifier or os.environ.get("USER", "unknown")

    print(f"Vendor:    {golden.get('vendor_key')}")
    print(f"Type:      {golden.get('vendor_type')}")
    print(f"Source PDF: {golden.get('pdf_path')}")
    print(f"Total SKUs: {len(skus)}")
    print(f"Already verified: {sum(1 for s in skus if s.get('_verified'))}")
    print(f"Verifier:  {verifier}")
    print()
    print("Open the PDF at the page each SKU cites, then accept/edit/skip.")
    print("Ctrl-C at any time is safe — verified entries persist.")
    print()
    try:
        input("Press Enter to start, Ctrl-C to abort > ")
    except (EOFError, KeyboardInterrupt):
        return 0

    counts = {"accepted": 0, "edited": 0, "skipped": 0, "quit": 0}
    now = datetime.now(timezone.utc).isoformat()
    for i, entry in enumerate(skus):
        if args.skip_verified and entry.get("_verified"):
            continue
        action = verify_one(entry, i, len(skus))
        counts[action] = counts.get(action, 0) + 1
        if action in ("accepted", "edited"):
            entry["_verified"] = True
            entry["_verified_at"] = now
            entry["_verified_by"] = verifier
            save_progress(args.golden, golden)
        if action == "quit":
            break

    print()
    print("=" * 80)
    print("Session summary:")
    for action, n in counts.items():
        print(f"  {action}: {n}")
    verified_now = sum(1 for s in skus if s.get("_verified"))
    print(f"\nTotal verified in golden file: {verified_now} / {len(skus)}")
    print(f"Saved: {args.golden}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
