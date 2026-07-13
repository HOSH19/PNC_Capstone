"""Contract tests for the GDELT poller (the canonical poller template)."""

from datetime import UTC, datetime, timedelta

import pytest

from pipeline import poll_gdelt


class JsonResp:
    def __init__(self, articles):
        self._articles = articles

    def json(self):
        return {"articles": self._articles}


def art(url, title, seendate="20260710T120000Z"):
    return {"url": url, "title": title, "seendate": seendate, "domain": "x.com"}


def test_single_phrase_query_is_not_parenthesized(monkeypatch):
    seen = {}
    def fake_get(url, **kw):
        seen.update(kw["params"])
        return JsonResp([])
    monkeypatch.setattr(poll_gdelt, "throttled_get", fake_get)
    now = datetime.now(UTC)
    poll_gdelt.fetch_window('"Bank of America"', now - timedelta(hours=1), now)
    assert seen["query"] == '"Bank of America"'   # GDELT rejects parens without OR
    poll_gdelt.fetch_window('"PNC Financial" OR "PNC Bank"', now - timedelta(hours=1), now)
    assert seen["query"].startswith("(") and seen["query"].endswith(")")


def test_full_page_bisects_and_floor_returns_capped(monkeypatch, capsys):
    calls = []
    full = [art(f"u{i}", f"t{i}") for i in range(250)]
    monkeypatch.setattr(poll_gdelt, "throttled_get",
                        lambda url, **kw: calls.append(1) or JsonResp(full))
    start = datetime(2026, 7, 10, 12, 0, tzinfo=UTC)
    arts = poll_gdelt.fetch_window("q", start, start + timedelta(minutes=2))
    assert len(calls) == 3          # parent + two 1-minute halves
    assert len(arts) == 500         # each floor window returns its cap
    assert "WARNING" in capsys.readouterr().err   # the drop is loud, not silent


def test_to_rows_folds_syndication_and_dedups_urls():
    rows = poll_gdelt.to_rows("pnc", [
        art("https://a.com/1", "Bank Distress Story"),
        art("https://a.com/1", "Bank Distress Story"),          # same URL
        art("https://b.com/2", "Bank  Distress -- Story!"),     # same title normalized
        art("https://c.com/3", "Something Else Entirely"),
    ])
    assert len(rows) == 2
    assert rows[0]["n_duplicates"] == 1


def test_one_bad_bank_does_not_starve_the_rest(monkeypatch, fake_db):
    fake_db.banks = [{"bank_id": b, "gdelt_query": f"q-{b}"} for b in ("bac", "citi", "jpm")]
    def fetch(query, start, end):
        if "citi" in query:
            raise RuntimeError("GDELT non-JSON response: malformed")
        return [art(f"https://x.com/{query}", f"story {query}")]
    monkeypatch.setattr(poll_gdelt, "fetch_window", fetch)

    with pytest.raises(SystemExit, match="citi"):
        poll_gdelt.main()

    polled = {k[1] for k in fake_db.watermarks}
    assert polled == {"bac", "jpm"}              # citi's watermark did not advance
    assert fake_db.heartbeats[-1][3] is False    # partial failure is not silent


def test_overlap_rerun_inserts_nothing_new(monkeypatch, fake_db):
    fake_db.banks = [{"bank_id": "pnc", "gdelt_query": "q"}]
    batches = iter([
        [art("https://a.com/x", "Big Bank Story")],
        [art("https://c.com/x", "BIG BANK STORY!"),   # new syndicated URL, same title
         art("https://a.com/x", "Big Bank Story")],
    ])
    monkeypatch.setattr(poll_gdelt, "fetch_window", lambda q, s, e: next(batches))
    poll_gdelt.main()
    poll_gdelt.main()
    assert len(fake_db.rows) == 1
    assert fake_db.heartbeats[1][2] == 0   # second run inserted zero
