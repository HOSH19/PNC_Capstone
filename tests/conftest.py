"""Shared fixtures: an in-memory stand-in for pipeline.db.

Tests never touch Postgres or the network — they monkeypatch the real
pipeline.db module's functions so poller code runs unchanged.
"""

import types

import pytest

from pipeline import db as real_db


class FakeDB:
    def __init__(self):
        self.rows = []          # everything passed to upsert_raw_items
        self.watermarks = {}    # (source, bank_id) -> datetime
        self.heartbeats = []    # (job, seen, inserted, ok)
        self.banks = []
        self.rollbacks = 0

    def install(self, monkeypatch):
        conn = types.SimpleNamespace(
            rollback=lambda: setattr(self, "rollbacks", self.rollbacks + 1),
            close=lambda: None,
        )
        monkeypatch.setattr(real_db, "connect", lambda: conn)
        monkeypatch.setattr(real_db, "get_live_banks", lambda c: self.banks)
        monkeypatch.setattr(real_db, "get_watermark",
                            lambda c, s, b: self.watermarks.get((s, b)))
        monkeypatch.setattr(real_db, "set_watermark",
                            lambda c, s, b, t: self.watermarks.__setitem__((s, b), t))
        monkeypatch.setattr(real_db, "upsert_raw_items", self._upsert)
        monkeypatch.setattr(real_db, "existing_title_hashes",
                            lambda c, s, b, hs: {h for h in hs if h in self._title_hashes()})
        monkeypatch.setattr(real_db, "write_heartbeat",
                            lambda c, job, seen, ins, dur, ok: self.heartbeats.append(
                                (job, seen, ins, ok)))
        return self

    def _title_hashes(self):
        return {r["title_hash"] for r in self.rows if r.get("title_hash")}

    def _upsert(self, conn, rows):
        seen = {(r["source"], r["external_id"], r["bank_id"]) for r in self.rows}
        inserted = 0
        for r in rows:
            key = (r["source"], r["external_id"], r["bank_id"])
            if key not in seen:
                self.rows.append(r)
                seen.add(key)
                inserted += 1
        return inserted


@pytest.fixture
def fake_db(monkeypatch):
    return FakeDB().install(monkeypatch)
