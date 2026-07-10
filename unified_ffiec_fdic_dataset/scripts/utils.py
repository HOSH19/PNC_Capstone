"""Shared utility helpers for parsing, IO, and numeric transforms."""

import csv
from datetime import datetime


def write_csv(path, header, rows):
    """Write rows to a CSV file atomically."""
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with open(temp_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    temp_path.replace(path)


def read_csv_rows(path):
    """Return all rows from a CSV if it exists."""
    if not path.exists():
        return []
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def iso_from_repdte(value):
    """Convert YYYYMMDD into YYYY-MM-DD."""
    if not value:
        return ""
    text = str(value)
    return f"{text[0:4]}-{text[4:6]}-{text[6:8]}"


def iso_from_mdy(value):
    """Convert M/D/YYYY-style strings into ISO dates."""
    if not value:
        return ""
    for fmt in ("%m/%d/%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(str(value).strip(), fmt).date().isoformat()
        except ValueError:
            pass
    return ""


def numeric(value):
    """Convert numeric-like text to float, preserving blanks as None."""
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def integer(value):
    """Convert numeric-like text to int, preserving blanks as None."""
    number = numeric(value)
    return int(number) if number is not None else None


def numeric_sum(*values):
    """Sum non-null numbers and return None if all are missing."""
    present = [value for value in values if value is not None]
    return sum(present) if present else None


def first_value(record, keys):
    """Return the first populated field from a parsed schedule row."""
    for key in keys:
        value = numeric(record.get(key))
        if value is not None:
            return value
    return None


def percentage_ratio(numerator, denominator):
    """Return a percentage ratio rounded for storage."""
    if numerator is None or denominator in (None, 0):
        return None
    return round(numerator / denominator * 100, 4)


def quarter_sequence(start_year, end_quarter):
    """Return quarter-end dates from start_year through end_quarter."""
    quarter_endings = ("0331", "0630", "0930", "1231")
    quarters = []
    for year in range(start_year, int(end_quarter[:4]) + 1):
        for suffix in quarter_endings:
            repdte = f"{year}{suffix}"
            if repdte <= end_quarter:
                quarters.append(repdte)
    return list(reversed(quarters))
