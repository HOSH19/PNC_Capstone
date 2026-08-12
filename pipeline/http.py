"""Shared throttled HTTP GET with retry/backoff for pollers.

Source-specific behavior stays in the caller as parameters (headers,
retry_statuses) so pollers never fork this logic — copy the main()
skeleton from poll_gdelt.py, not this file. See RUNBOOK.md §6.

The rate limiter is module-global: at most one request per THROTTLE_S
per process. Each poller runs as its own process, so the throttle is
effectively per source.
"""

import time

import requests

THROTTLE_S = 1.0
MAX_RETRIES = 5

_last_request_at = 0.0


class RetriesExhausted(RuntimeError):
    """The API kept returning a retryable status until the budget ran out.

    Subclasses RuntimeError so existing `except RuntimeError` callers are
    unaffected; it exists so a caller that can simply come back later --
    the backfill, which resumes from a watermark -- can tell "the API is
    rate-limiting us" apart from "this request is broken".
    """

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def throttled_get(
    url: str,
    *,
    params: dict | None = None,
    headers: dict | None = None,
    retry_statuses: tuple = (429,),
    throttle_s: float = THROTTLE_S,
    label: str = "",
) -> requests.Response:
    """GET with request spacing, exponential backoff, and Retry-After.

    Retries retry_statuses and all 5xx; raises RetriesExhausted (a
    RuntimeError) when the budget runs out, requests.HTTPError for any
    other non-2xx (e.g. 404). throttle_s overrides the request spacing for
    stricter APIs — GDELT now needs tens of seconds, see poll_gdelt.
    """
    global _last_request_at
    resp = None
    for attempt in range(MAX_RETRIES):
        wait = throttle_s - (time.monotonic() - _last_request_at)
        if wait > 0:
            time.sleep(wait)
        _last_request_at = time.monotonic()
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=60)
        except requests.RequestException:
            # Timeouts and connection resets deserve the same retry budget
            # as a 5xx (GDELT gets slow under load).
            if attempt + 1 >= MAX_RETRIES:
                raise
            time.sleep(max(1.0, throttle_s) * 2**attempt)
            continue
        if resp.status_code in retry_statuses or resp.status_code >= 500:
            if attempt + 1 < MAX_RETRIES:  # no pointless sleep before giving up
                # Scale backoff by the API's own pacing: 1/2/4/8 s for SEC,
                # 5/10/20/40 s for GDELT, whose 429s outlast short waits.
                delay = max(1.0, throttle_s) * 2**attempt
                retry_after = resp.headers.get("Retry-After", "")
                if retry_after.isdigit():
                    delay = max(delay, int(retry_after))
                time.sleep(delay)
            continue
        resp.raise_for_status()
        return resp
    raise RetriesExhausted(
        f"{label or url} still failing after {MAX_RETRIES} retries: {resp.status_code}",
        status=resp.status_code,
    )
