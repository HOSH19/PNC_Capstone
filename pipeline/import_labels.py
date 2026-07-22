"""Import a Kaggle labels CSV into item_label (Stage 1 back half).

Reads labels_<date>.csv (raw_item_id, label, model_meta) and upserts one row
per (raw_item_id, label_source). ON CONFLICT overwrites only THIS labeler's
row, so re-labeling with a new prompt replaces its own history and never
touches other labelers' rows (DESIGN Kaggle round-trip).

Validation is all-or-nothing: a single bad row aborts before any DB write.
Label validation reuses pipeline.labeling.validate_label (one definition).
"""

import argparse
import csv
import json
import sys

from psycopg.types.json import Jsonb

from pipeline import db
from pipeline.labeling import validate_label

# Mirrors the label_source CHECK in db/migrations/011_scoring_tables.sql.
VALID_LABEL_SOURCES = ("llama_kaggle", "gemini", "human")


def parse_labels(csv_rows: list[dict]) -> list[dict]:
    """Validate CSV rows into upsert records. Pure, no DB.

    Returns [{raw_item_id: int, label: str, model_meta: dict}, ...].
    Raises ValueError listing every bad row (nothing is returned partially).
    """
    records: list[dict] = []
    bad: list[str] = []
    for i, r in enumerate(csv_rows):
        try:
            meta_raw = r.get("model_meta") or ""
            records.append(
                {
                    "raw_item_id": int(r["raw_item_id"]),
                    "label": validate_label(r["label"]),
                    "model_meta": json.loads(meta_raw) if meta_raw else {},
                }
            )
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
            bad.append(f"row {i}: {e}")
    if bad:
        raise ValueError("invalid label rows:\n" + "\n".join(bad))
    return records


def upsert_labels(conn, records: list[dict], label_source: str) -> int:
    """Upsert records for one labeler. Returns count sent."""
    if not records:
        return 0
    sql = (
        "INSERT INTO item_label (raw_item_id, label, label_source, model_meta) "
        "VALUES (%(raw_item_id)s, %(label)s, %(label_source)s, %(model_meta)s) "
        "ON CONFLICT (raw_item_id, label_source) DO UPDATE "
        "SET label = EXCLUDED.label, "
        "    model_meta = EXCLUDED.model_meta, "
        "    labeled_at = now()"
    )
    payload = [
        {**rec, "label_source": label_source, "model_meta": Jsonb(rec["model_meta"])}
        for rec in records
    ]
    with conn.cursor() as cur:
        cur.executemany(sql, payload)
    conn.commit()
    return len(records)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", help="labels_<date>.csv")
    ap.add_argument("--label-source", default="llama_kaggle")
    args = ap.parse_args()
    if args.label_source not in VALID_LABEL_SOURCES:
        sys.exit(
            f"unknown --label-source {args.label_source!r}; "
            f"allowed: {VALID_LABEL_SOURCES}"
        )

    with open(args.csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    records = parse_labels(rows)  # raises before any DB write if malformed

    conn = db.connect()
    try:
        n = upsert_labels(conn, records, args.label_source)
    finally:
        conn.close()
    print(f"upserted {n} labels as label_source={args.label_source}", file=sys.stderr)


if __name__ == "__main__":
    main()
