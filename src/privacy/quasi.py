"""Quasi-identifier protection: per-record keyed date shifting.

Why per-patient and not global: the published attacker recovers a global offset from two known
dates and then re-identifies everyone. A keyed per-patient offset makes every patient's shift
independent, so one recovery says nothing about the next patient.

Why shifting instead of redaction: shifting *all* of a record's dates by one constant preserves
every interval - length of stay, drug course, lab-to-discharge gaps - which is exactly what the
utility probe measures. Redaction would destroy that.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import re

_MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]

# Date shapes that appear in the corpus: dd/mm/yyyy, dd-mm-yyyy, yyyy-mm-dd, "5 September 2026".
_SLASH = re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b")
_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_NAMED = re.compile(r"\b(\d{1,2})\s+(" + "|".join(_MONTHS) + r")[a-z]*\s+(\d{4})\b", re.IGNORECASE)


def offset_days(secret: str | None, key: str) -> int:
    """Deterministic per-key offset in [-180, +180] days, never zero."""
    if not secret:
        return 0
    digest = hmac.new(secret.encode(), f"dateshift|{key}".encode(), hashlib.sha256).digest()
    value = int.from_bytes(digest[:4], "big") % 361
    return value - 180 or 1


def _shift_date(day: int, month: int, year: int, offset: int) -> tuple[int, int, int] | None:
    try:
        shifted = dt.date(year, month, day) + dt.timedelta(days=offset)
    except ValueError:
        return None
    return shifted.day, shifted.month, shifted.year


def shift_text(text: str, offset: int) -> str:
    if not offset or not text:
        return text

    def slash(match: re.Match) -> str:
        d, m, y = (int(g) for g in match.groups())
        shifted = _shift_date(d, m, y, offset)
        if not shifted:
            return match.group(0)
        sep = match.group(0)[len(str(match.group(1)))]
        return f"{shifted[0]:02d}{sep}{shifted[1]:02d}{sep}{shifted[2]}"

    def iso(match: re.Match) -> str:
        y, m, d = (int(g) for g in match.groups())
        shifted = _shift_date(d, m, y, offset)
        return f"{shifted[2]}-{shifted[1]:02d}-{shifted[0]:02d}" if shifted else match.group(0)

    def named(match: re.Match) -> str:
        d, mon, y = int(match.group(1)), match.group(2).lower()[:3], int(match.group(3))
        shifted = _shift_date(d, _MONTHS.index(mon) + 1, y, offset)
        return f"{shifted[0]} {match.group(2)} {shifted[2]}" if shifted else match.group(0)

    return _NAMED.sub(named, _SLASH.sub(slash, _ISO.sub(iso, text)))


def shift_ir(ir, offset: int) -> int:
    """Apply the offset to every date-shaped string in the record. Returns replacements made."""
    count = 0

    def walk(node, depth=0):
        nonlocal count
        if depth > 24:
            return
        if isinstance(node, dict):
            for key, value in list(node.items()):
                if isinstance(value, str) and value:
                    new = shift_text(value, offset)
                    if new != value:
                        node[key] = new
                        count += 1
                else:
                    walk(value, depth + 1)
        elif isinstance(node, list):
            for index, value in enumerate(node):
                if isinstance(value, str) and value:
                    new = shift_text(value, offset)
                    if new != value:
                        node[index] = new
                        count += 1
                else:
                    walk(value, depth + 1)

    data = getattr(getattr(ir, "structure", None), "data", None)
    if isinstance(data, str):
        new = shift_text(data, offset)
        if new != data:
            ir.structure.data = new
            count += 1
    else:
        walk(data)
    return count


# ── age banding and place generalisation ───────────────────────────────────
_AGE_KEY = re.compile(r"^(?:age|age_?years?|patient_?age|वय)$", re.IGNORECASE)
_AGE_SEX = re.compile(r"^(\d{1,3})\s*/\s*([MFmf])$")
_VILLAGE_KEY = re.compile(r"(?:village|gram|mohalla|locality|town)", re.IGNORECASE)
_DISTRICT_KEY = re.compile(r"(?:district|distt|dist\.)", re.IGNORECASE)


def band_age(age: int) -> str:
    """5-year band whose floor is <= the true age, so the truth always falls inside its band."""
    low = (max(0, min(age, 120)) // 5) * 5
    return f"{low}-{low + 4}"


def band_ages(ir) -> int:
    """Band integer ages (and 'age/sex' pairs) in structured records only."""
    changed = 0

    def walk(node, hint="", depth=0):
        nonlocal changed
        if depth > 24:
            return
        if isinstance(node, dict):
            for key, value in list(node.items()):
                key = str(key)
                if _AGE_KEY.match(key.strip()) and isinstance(value, int) and 0 <= value <= 120:
                    node[key] = band_age(value)
                    changed += 1
                elif _AGE_KEY.match(key.strip()) and isinstance(value, str):
                    match = _AGE_SEX.match(value.strip())
                    if match:
                        node[key] = f"{band_age(int(match.group(1)))}/{match.group(2)}"
                        changed += 1
                    elif value.strip().isdigit() and len(value.strip()) <= 3:
                        node[key] = band_age(int(value.strip()))
                        changed += 1
                else:
                    if isinstance(value, str):
                        match = _AGE_SEX.match(value.strip())
                        if match and "age" in key.lower():
                            node[key] = f"{band_age(int(match.group(1)))}/{match.group(2)}"
                            changed += 1
                    else:
                        walk(value, key, depth + 1)
        elif isinstance(node, list):
            for index, value in enumerate(node):
                if isinstance(value, str):
                    match = _AGE_SEX.match(value.strip())
                    if match:
                        node[index] = f"{band_age(int(match.group(1)))}/{match.group(2)}"
                        changed += 1
                else:
                    walk(value, hint, depth + 1)

    data = getattr(getattr(ir, "structure", None), "data", None)
    walk(data)
    return changed


def build_place_map(root) -> dict[str, str]:
    """village -> district, harvested from the corpus's own registration files (no hardcoding)."""
    import csv
    from pathlib import Path

    mapping: dict[str, str] = {}
    for path in sorted(Path(root).rglob("*.csv")):
        try:
            rows = list(csv.reader(open(path, encoding="utf-8", errors="ignore")))
        except OSError:
            continue
        if not rows:
            continue
        header = [h.strip() for h in rows[0]]
        village = next((i for i, h in enumerate(header) if _VILLAGE_KEY.search(h)), None)
        district = next((i for i, h in enumerate(header) if _DISTRICT_KEY.search(h)), None)
        if village is None or district is None:
            continue
        for row in rows[1:]:
            if max(village, district) < len(row):
                name, place = row[village].strip(), row[district].strip()
                if name and place and len(name) > 2:
                    mapping.setdefault(name.lower(), place)
    return mapping


def generalise_places(ir, mapping: dict[str, str]) -> int:
    """Replace a village with its district; leave anything not in the mapping alone."""
    if not mapping:
        return 0
    changed = 0

    def walk(node, hint="", depth=0):
        nonlocal changed
        if depth > 24:
            return
        if isinstance(node, dict):
            for key, value in list(node.items()):
                key = str(key)
                if isinstance(value, str):
                    replacement = mapping.get(value.strip().lower())
                    if replacement and (_VILLAGE_KEY.search(key) or len(value.split()) <= 3):
                        node[key] = replacement
                        changed += 1
                else:
                    walk(value, key, depth + 1)
        elif isinstance(node, list):
            for index, value in enumerate(node):
                if isinstance(value, str):
                    replacement = mapping.get(value.strip().lower())
                    if replacement:
                        node[index] = replacement
                        changed += 1
                else:
                    walk(value, hint, depth + 1)

    walk(getattr(getattr(ir, "structure", None), "data", None))
    return changed
