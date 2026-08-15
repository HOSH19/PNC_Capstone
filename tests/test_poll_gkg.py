from datetime import UTC, datetime, timedelta

from pipeline.poll_gkg import FIRST_RUN_LOOKBACK, OVERLAP, window_start

NOW = datetime(2026, 8, 15, 12, 0, tzinfo=UTC)


def test_first_run_looks_back_a_fixed_window():
    assert window_start([None, None], NOW) == NOW - FIRST_RUN_LOOKBACK


def test_window_follows_the_oldest_watermark_not_the_newest():
    """One query serves every bank, so a bank left behind would be skipped
    entirely if the window tracked the newest mark."""
    marks = [NOW - timedelta(days=4), NOW - timedelta(hours=2), None]
    assert window_start(marks, NOW) == NOW - timedelta(days=4) - OVERLAP


def test_overlap_re_reads_a_partition_that_is_still_filling():
    """GKG partitions land through the day; ending a window at the newest
    watermark would drop rows that arrived after it."""
    marks = [NOW - timedelta(hours=1)]
    assert window_start(marks, NOW) < NOW - timedelta(hours=1)
