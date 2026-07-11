"""Thin Postgres client for the ingestion pipeline. Plain SQL, no ORM.

Connection comes from SUPABASE_DB_URL (Supabase session-pooler DSN).
"""

import os
from datetime import datetime

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

RAW_ITEM_COLUMNS = (
    "source", "external_id", "bank_id", "published_at", "title", "url",
    "domain", "text_excerpt", "title_hash", "n_duplicates", "meta",
)


def connect() -> psycopg.Connection:
    return psycopg.connect(os.environ["SUPABASE_DB_URL"], row_factory=dict_row)


def get_live_banks(conn) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM bank WHERE is_live ORDER BY bank_id")
        return cur.fetchall()


def upsert_raw_items(conn, rows: list[dict]) -> int:
    """Batch insert, ignoring rows already present. Returns rows actually inserted."""
    if not rows:
        return 0
    cols = ", ".join(RAW_ITEM_COLUMNS)
    params = ", ".join(f"%({c})s" for c in RAW_ITEM_COLUMNS)
    payload = [
        {**{c: r.get(c) for c in RAW_ITEM_COLUMNS}, "meta": Jsonb(r.get("meta") or {})}
        for r in rows
    ]
    with conn.cursor() as cur:
        cur.executemany(
            f"INSERT INTO raw_item ({cols}) VALUES ({params}) "
            "ON CONFLICT (source, external_id, bank_id) DO NOTHING",
            payload,
        )
        inserted = cur.rowcount
    conn.commit()
    return inserted


def get_watermark(conn, source: str, bank_id: str) -> datetime | None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT last_polled_at FROM watermark WHERE source = %s AND bank_id = %s",
            (source, bank_id),
        )
        row = cur.fetchone()
        return row["last_polled_at"] if row else None


def set_watermark(conn, source: str, bank_id: str, last_polled_at: datetime) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO watermark (source, bank_id, last_polled_at) VALUES (%s, %s, %s) "
            "ON CONFLICT (source, bank_id) DO UPDATE SET last_polled_at = EXCLUDED.last_polled_at",
            (source, bank_id, last_polled_at),
        )
    conn.commit()


def write_heartbeat(conn, job: str, items_seen: int, items_inserted: int,
                    duration_s: float, ok: bool) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO pipeline_heartbeat (job, items_seen, items_inserted, duration_s, ok) "
            "VALUES (%s, %s, %s, %s, %s)",
            (job, items_seen, items_inserted, duration_s, ok),
        )
    conn.commit()
