"""Contract tests for the shared throttled HTTP helper."""

import time as real_time
import types

import pytest
import requests

from pipeline import http as ph


class Resp:
    def __init__(self, status, retry_after=None):
        self.status_code = status
        self.headers = {"Retry-After": retry_after} if retry_after else {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")


@pytest.fixture
def wired(monkeypatch):
    """Route sleeps into a list and requests.get through a scripted queue."""
    sleeps = []
    monkeypatch.setattr(ph, "time", types.SimpleNamespace(
        monotonic=real_time.monotonic, sleep=sleeps.append))

    def run(effects, **kw):
        it = iter(effects)
        def get(*a, **k):
            e = next(it)
            if isinstance(e, Exception):
                raise e
            return e
        monkeypatch.setattr(ph, "requests", types.SimpleNamespace(
            get=get, RequestException=requests.RequestException))
        return ph.throttled_get("https://x", **kw)

    return run, sleeps


def backoffs(sleeps, floor=1.0):
    return [s for s in sleeps if s >= floor]


def test_retry_after_is_honored(wired):
    run, sleeps = wired
    run([Resp(429, "30"), Resp(200)])
    assert 30 in backoffs(sleeps)


def test_backoff_scales_with_throttle(wired):
    run, sleeps = wired
    with pytest.raises(RuntimeError):
        run([Resp(429)] * 5, retry_statuses=(429,), throttle_s=5.0, label="GDELT")
    assert backoffs(sleeps, 5) == [5, 10, 20, 40]  # and no pointless final sleep


def test_sec_backoff_unchanged(wired):
    run, sleeps = wired
    with pytest.raises(RuntimeError):
        run([Resp(500)] * 5, throttle_s=1.0)
    assert backoffs(sleeps) == [1, 2, 4, 8]


def test_timeouts_use_the_retry_budget(wired):
    run, sleeps = wired
    resp = run([requests.ReadTimeout("t"), requests.ConnectionError("c"), Resp(200)],
               throttle_s=5.0)
    assert resp.status_code == 200


def test_persistent_timeout_raises(wired):
    run, _ = wired
    with pytest.raises(requests.ReadTimeout):
        run([requests.ReadTimeout("t")] * 5)


def test_non_retryable_4xx_raises_immediately(wired):
    run, sleeps = wired
    with pytest.raises(requests.HTTPError):
        run([Resp(404)])
    assert backoffs(sleeps) == []
