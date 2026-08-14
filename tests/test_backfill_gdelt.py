"""Unit tests for the GDELT backfill's pure logic: windowing, job chunking,
resume. No DB, no network — the API-shaped parts are poll_gdelt's and are
covered by test_poll_gdelt.py."""

from datetime import UTC, datetime, timedelta

import pytest

from pipeline import backfill_gdelt as bf


def d(y, m, day=1):
    return datetime(y, m, day, tzinfo=UTC)


def test_windows_tile_the_range_without_gaps_or_overlap():
    start, end = d(2020, 1), d(2025, 1)
    windows = bf.iter_windows(start, end)
    assert windows[0][0] == start
    assert windows[-1][1] == end  # last window is truncated, not overshot
    assert all(a[1] == b[0] for a, b in zip(windows, windows[1:], strict=False))
    assert all(s < e <= end for s, e in windows)
    # ~20 quarterly windows/bank x 104 banks is the 15 h estimate the
    # chunking is sized against; a silent change here changes the run length.
    assert len(windows) == 21


def test_partial_last_window_is_kept():
    windows = bf.iter_windows(d(2020, 1), d(2020, 2), timedelta(days=20))
    assert windows == [(d(2020, 1), d(2020, 1, 21)), (d(2020, 1, 21), d(2020, 2))]


def test_slices_partition_banks_exactly_once():
    banks = [f"b{i:03d}" for i in range(104)]
    slices = [bf.bank_slice(banks, i, 4) for i in range(4)]
    assert sorted(sum(slices, [])) == banks  # every bank, no duplicates
    assert {len(s) for s in slices} == {26}
    with pytest.raises(ValueError):
        bf.bank_slice(banks, 4, 4)


def test_resume_skips_committed_windows_and_finished_banks():
    start, end = d(2020, 1), d(2025, 1)
    assert bf.resume_start(None, start, end) == start  # never run
    assert bf.resume_start(d(2019, 6), start, end) == start  # older than range
    assert bf.resume_start(d(2022, 4), start, end) == d(2022, 4)  # mid-flight
    assert bf.resume_start(end, start, end) is None  # done
    assert bf.resume_start(d(2026, 4), start, end) is None  # past the range


def test_resume_after_a_kill_refetches_nothing_already_committed():
    start, end = d(2020, 1), d(2021, 1)
    done = bf.iter_windows(start, end)[:2]  # two windows survived
    remaining = bf.iter_windows(bf.resume_start(done[-1][1], start, end), end)
    assert done[-1][1] == remaining[0][0]  # resumes at the seam
    assert bf.iter_windows(start, end) == done + remaining


def test_backfill_watermark_does_not_collide_with_the_live_poller():
    """Writing 2021 into ('gdelt', bank) would rewind live polling 5 years."""
    from pipeline import poll_gdelt

    assert bf.WATERMARK_SOURCE != "gdelt"
    row = poll_gdelt.to_rows("pnc", [{"url": "u", "title": "t"}])[0]
    assert row["source"] == "gdelt"  # rows still land in the one gdelt source


def test_request_spacing_stays_above_the_rate_that_429s():
    """8 s is measured to fail: every scheduled poll from 2026-08-11 died on
    429 at that spacing. Neither caller may drift back down to it.

    The two are set independently on purpose — the poller makes ~104 requests
    per run and can afford to be slow, while spacing multiplies the backfill's
    thousands — so this pins the floor, not the values.
    """
    from pipeline import poll_gdelt

    KNOWN_TOO_FAST = 8.0
    assert bf.THROTTLE_S > KNOWN_TOO_FAST
    assert poll_gdelt.THROTTLE_S > KNOWN_TOO_FAST


def test_rate_limit_is_incomplete_not_failure():
    """Being 429'd is the one condition the resume design exists to absorb.

    Treating it as a failure paints every pass red on the exact signal that
    means "come back later", which is what `incomplete` already reports.
    """
    from pipeline.http import RetriesExhausted

    assert issubclass(RetriesExhausted, RuntimeError)  # old callers unaffected
    exc = RetriesExhausted("GDELT still failing after 5 retries: 429", status=429)
    assert exc.status == 429


def test_truncated_json_is_transient_but_a_text_error_is_not():
    """A body that starts like JSON and fails to parse is the API buckling;
    a plain-text error is a malformed query that will fail forever. Only the
    first should let the backfill shrug and resume next run."""
    from pipeline import poll_gdelt
    from pipeline.http import Transient

    class FakeResp:
        def __init__(self, text):
            self.text = text

        def json(self):
            raise ValueError("no")

    def fetch(text):
        poll_gdelt.throttled_get = lambda *a, **kw: FakeResp(text)
        try:
            poll_gdelt.fetch_window(
                "q",
                datetime(2020, 1, 1, tzinfo=UTC),
                datetime(2020, 1, 2, tzinfo=UTC),
            )
        except Exception as exc:
            return exc

    real = poll_gdelt.throttled_get
    try:
        assert isinstance(fetch('{"articles": [ {"url": "x"'), Transient)
        err = fetch("ERROR: malformed query")
        assert isinstance(err, RuntimeError) and not isinstance(err, Transient)
    finally:
        poll_gdelt.throttled_get = real
