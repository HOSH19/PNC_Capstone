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


def throttled_get(url: str, *, params: dict | None = None,
                  headers: dict | None = None,
                  retry_statuses: tuple = (429,),
                  label: str = "") -> requests.Response:
    """GET with request spacing, exponential backoff, and Retry-After.

    Retries retry_statuses and all 5xx; raises RuntimeError when retries
    are exhausted, requests.HTTPError for any other non-2xx (e.g. 404).
    """
    global _last_request_at
    resp = None
    for attempt in range(MAX_RETRIES):
        wait = THROTTLE_S - (time.monotonic() - _last_request_at)
        if wait > 0:
            time.sleep(wait)
        _last_request_at = time.monotonic()
        resp = requests.get(url, params=params, headers=headers, timeout=60)
        if resp.status_code in retry_statuses or resp.status_code >= 500:
            if attempt + 1 < MAX_RETRIES:  # no pointless sleep before giving up
                delay = 2 ** attempt
                retry_after = resp.headers.get("Retry-After", "")
                if retry_after.isdigit():
                    delay = max(delay, int(retry_after))
                time.sleep(delay)
            continue
        resp.raise_for_status()
        return resp
    raise RuntimeError(
        f"{label or url} still failing after {MAX_RETRIES} retries: {resp.status_code}")
