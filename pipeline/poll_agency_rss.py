"""Incremental poller: Fed general press-release/policy RSS -> raw_item.

Unlike poll_gdelt.py's per-bank query loop, RSS feeds are not queryable by
bank -- each feed just returns its latest ~15-20 items regardless of
content. This poller fetches each feed ONCE per run, then scans every
item's title+summary against every live bank's aliases; a match emits one
row per (item, bank) pair. An item can match zero, one, or several banks;
items matching none are skipped (expected -- most Fed releases are not
about a tracked bank).

No watermark: these feeds have no date-range query capability (confirmed
by hand -- they only ever return the current latest window), so there is
no "window" to advance. Re-fetching the same latest items every run is
expected; UNIQUE(source, external_id, bank_id) makes repeat inserts free
no-ops, same as the enforcement pollers.

Distinct from poll_fed_enforcement.py: that covers enforcement actions
from a dedicated CSV export. This covers general press releases and
policy notices, which enforcement actions are not a subset of.

Run: python -m pipeline.poll_agency_rss
"""

import hashlib
import re
import sys
import time
from datetime import UTC, datetime

import feedparser

from pipeline import db
from pipeline.http import throttled_get

FEEDS = {
    "fed_all_releases": "https://www.federalreserve.gov/feeds/press_all.xml",
    "fed_bank_reg_policy": "https://www.federalreserve.gov/feeds/press_bcreg.xml",
    "fed_h41": "https://www.federalreserve.gov/feeds/h41.xml",
}


def normalize_title_hash(title: str) -> str:
    normalized = re.sub(r"[\W_]+", " ", title.lower()).strip()
    return hashlib.sha1(normalized.encode()).hexdigest()


def build_alias_index(banks: list[dict]) -> list[tuple[str, list[re.Pattern]]]:
    """[(bank_id, [word-boundary regexes for legal_name/holding_name/aliases])].

    Word-boundary, not plain substring: a bare short alias like "BNY" is a
    substring of unrelated text ("Federal Reserve Bank of New York (FRBNY)"
    contains "bny"). \\b anchors require the alias to appear as a whole word,
    which a substring scan does not -- confirmed against real Fed H.4.1 text
    during testing (see db/seed/banks.csv note on bny: "bare 'BNY' too short
    for GDELT", the same failure mode GDELT's query strings already avoid).

    Banks whose notes are flagged "generic" (22 of them, e.g. ffbc/ffin both
    legally named "First Financial Bank") use holding_name only, mirroring
    the same convention already used for their gdelt_query -- their
    bank_legal_name/aliases collide with unrelated banks or generic industry
    phrases ("community bank leverage ratio" matched cbu's "Community Bank"
    alias in testing) and are unsafe for free-text substring matching.
    """
    out = []
    for b in banks:
        if b.get("notes") and "generic" in b["notes"].lower():
            names = [b.get("holding_name")]
        else:
            names = [b.get("bank_legal_name"), b.get("holding_name")] + list(b.get("aliases") or [])
        patterns = [re.compile(r"\b" + re.escape(n.lower()) + r"\b") for n in names if n]
        if patterns:
            out.append((b["bank_id"], patterns))
    return out


def match_banks(text: str, index: list[tuple[str, list[re.Pattern]]]) -> set[str]:
    text_lower = text.lower()
    return {bank_id for bank_id, patterns in index if any(p.search(text_lower) for p in patterns)}


def _parse_published(entry) -> datetime | None:
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if not parsed:
        return None
    return datetime(*parsed[:6], tzinfo=UTC)


def fetch_feed(feed_key: str, url: str) -> list:
    # throttled_get does the actual network fetch (retry/backoff/rate limit);
    # feedparser only parses the bytes we already have, so it never makes
    # its own request (would bypass throttled_get if passed the URL directly).
    resp = throttled_get(url, label=feed_key, throttle_s=2.0)
    return feedparser.parse(resp.content).entries


def to_rows(feed_key: str, entries: list, index: list[tuple[str, list[str]]]) -> list[dict]:
    rows = []
    for e in entries:
        title = e.get("title") or ""
        summary = e.get("summary") or ""
        external_id = e.get("id") or e.get("link")
        if not external_id:
            continue
        matched = match_banks(f"{title} {summary}", index)
        if not matched:
            continue
        title_hash = normalize_title_hash(title or external_id)
        published_at = _parse_published(e)
        for bank_id in matched:
            rows.append({
                "source": "agency_rss",
                "external_id": external_id,
                "bank_id": bank_id,
                "published_at": published_at,
                "title": title,
                "url": e.get("link"),
                "domain": "federalreserve.gov",
                "text_excerpt": summary[:4000] if summary else None,
                "title_hash": title_hash,
                "n_duplicates": 0,
                "meta": {"feed": feed_key},
            })
    return rows


def main() -> None:
    started = time.monotonic()
    seen = inserted = 0
    failed: list[str] = []
    conn = db.connect()
    try:
        index = build_alias_index(db.get_live_banks(conn))
        for feed_key, url in FEEDS.items():
            try:
                entries = fetch_feed(feed_key, url)
                rows = to_rows(feed_key, entries, index)
                n = db.upsert_raw_items(conn, rows)
                seen += len(entries)
                inserted += n
                print(f"{feed_key}: {len(entries)} entries, {len(rows)} bank-matched, {n} inserted")
            except Exception as exc:
                # One bad feed must not stop the others.
                conn.rollback()
                failed.append(feed_key)
                print(f"{feed_key}: FAILED: {exc}", file=sys.stderr)
        db.write_heartbeat(conn, "poll_agency_rss", seen, inserted,
                           time.monotonic() - started, not failed)
    except Exception:
        try:
            conn.rollback()
            db.write_heartbeat(conn, "poll_agency_rss", seen, inserted,
                               time.monotonic() - started, False)
        except Exception:
            pass  # never mask the original failure
        raise
    finally:
        conn.close()
    if failed:
        sys.exit(f"poll_agency_rss: failed feeds: {', '.join(failed)}")


if __name__ == "__main__":
    sys.exit(main())
