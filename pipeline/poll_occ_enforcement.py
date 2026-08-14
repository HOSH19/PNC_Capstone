"""
OCC enforcement actions -> raw_item

Run:
    python -m pipeline.poll_occ_enforcement
"""

import re
import sys
import time
from datetime import datetime, timezone
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from pipeline import db
from pipeline.http import throttled_get
from pipeline.poll_fdic_enforcement import build_matcher


SEARCH_URL = "https://apps.occ.gov/EASearch/Search/Table"
BASE_URL = "https://apps.occ.gov"

PAGE_SIZE = 10
THROTTLE_S = 2.0
MAX_PAGES = 500  # ~5k rows/query; guard runaway pagination


def clean(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").replace("\xa0", " ").strip()


def parse_date(value: str | None) -> datetime | None:
    value = clean(value)

    if not value or value.upper() == "N/A":
        return None

    for fmt in (
        "%m/%d/%Y",
        "%Y-%m-%d",
        "%b %d, %Y",
        "%B %d, %Y",
    ):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue

    return None


def split_city_state(value: str | None) -> tuple[str, str]:
    value = clean(value)

    if "," not in value:
        return value, ""

    city, state = value.rsplit(",", 1)
    return city.strip(), state.strip()


def parse_result_row(row) -> dict | None:
    data = {}

    for td in row.select("td"):
        key = clean(td.get("data-th", ""))

        if not key:
            continue

        data[key] = clean(td.get_text(" ", strip=True))

        link = td.find("a", href=True)
        if link:
            data[f"{key}_url"] = urljoin(BASE_URL, link["href"])

    institution = data.get("Institution")

    if not institution:
        return None

    city, state = split_city_state(data.get("City, State", ""))
    url = data.get("Start Doc_url") or next(
        (v for k, v in data.items() if k.endswith("_url")),
        None,
    )

    return {
        "institution": institution,
        "charter_number": data.get("Charter Number"),
        "company_name": data.get("Company Name"),
        "individual": data.get("Individual"),
        "city": city,
        "state": state,
        "action_type": data.get("Type"),
        "amount": data.get("Amount"),
        "start_date": data.get("Start Date"),
        "termination_date": data.get("Termination Date"),
        "docket_number": data.get("Docket Number"),
        "subject": data.get("Subject Matters"),
        "url": url,
    }


def fetch_page(query: str, page: int) -> tuple[list[dict], int | None]:
    response = throttled_get(
        SEARCH_URL,
        label="OCC",
        throttle_s=THROTTLE_S,
        params={
            "q": query,
            "acs": "",
            "cat": "",
            "srt": "1",
            "pg": page,
            "pgsz": PAGE_SIZE,
            "isAdv": "false",
        },
        headers={"User-Agent": "Mozilla/5.0"},
    )

    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    page_text = clean(soup.get_text(" ", strip=True))

    match = re.search(
        r"of\s+([\d,]+)\s+total\s+records",
        page_text,
        re.IGNORECASE,
    )

    total = int(match.group(1).replace(",", "")) if match else None
    actions = []

    for row in soup.select("tr.show-results"):
        action = parse_result_row(row)

        if action:
            actions.append(action)

    return actions, total


def fetch_all(query: str) -> list[dict]:
    results = []
    page = 0
    total = None
    prev_page_keys: tuple[str, ...] | None = None

    while True:
        page_results, page_total = fetch_page(query, page)

        if total is None:
            total = page_total

        if not page_results:
            break

        page_keys = tuple(sorted(external_id(a) or "" for a in page_results))
        if prev_page_keys is not None and page_keys == prev_page_keys:
            print(
                f"WARNING: pagination repeated page {page} for {query!r}",
                file=sys.stderr,
            )
            break
        prev_page_keys = page_keys

        results.extend(page_results)

        if total is not None and len(results) >= total:
            break

        page += 1
        if page >= MAX_PAGES:
            print(
                f"WARNING: hit MAX_PAGES={MAX_PAGES} for {query!r}",
                file=sys.stderr,
            )
            break

    return results


def external_id(action: dict) -> str | None:
    """Stable OCC natural key; None when the row cannot be idempotently keyed."""
    parts: list[str] = []
    if action.get("docket_number"):
        parts.append(action["docket_number"])
    if action.get("institution"):
        parts.append(action["institution"])
    if action.get("individual"):
        parts.append(action["individual"])
    elif action.get("url"):
        parts.append(action["url"])
    elif action.get("action_type") and action.get("start_date"):
        parts.append(f"{action['action_type']}|{action['start_date']}")

    key = "|".join(parts)
    return key or None


def action_published_at(action: dict) -> datetime | None:
    return parse_date(action.get("start_date")) or parse_date(
        action.get("termination_date")
    )


def to_row(bank_id: str, action: dict) -> dict | None:
    eid = external_id(action)
    published_at = action_published_at(action)
    if not eid or published_at is None:
        return None

    return {
        "source": "occ_enforcement",
        "external_id": eid,
        "bank_id": bank_id,
        "published_at": published_at,
        "title": (
            f"{action['institution']} — "
            f"{action['action_type'] or 'OCC Enforcement Action'}"
        ),
        "url": action["url"],
        "domain": "apps.occ.gov",
        "text_excerpt": action["subject"] or None,
        "title_hash": None,
        "n_duplicates": 0,
        "meta": {
            key: value
            for key, value in action.items()
            if key not in {"institution", "url"}
        },
    }


def bank_queries(bank: dict) -> list[str]:
    values = [
        bank.get("bank_legal_name"),
        bank.get("holding_name"),
        *(bank.get("aliases") or []),
    ]

    seen = set()
    queries = []

    for value in values:
        value = clean(value)

        if value and value.lower() not in seen:
            seen.add(value.lower())
            queries.append(value)

    return queries


def main() -> None:
    started = time.monotonic()
    seen = 0
    inserted = 0
    failed = []

    conn = db.connect()

    try:
        match = build_matcher(db.get_live_banks(conn))
        for bank in db.get_live_banks(conn):
            bank_id = bank["bank_id"]

            try:
                actions = {}
                skipped = 0

                for query in bank_queries(bank):
                    print(f"Searching OCC: {query}")

                    for action in fetch_all(query):
                        if match(action["institution"]) != bank_id:
                            continue
                        key = external_id(action)
                        if key is None or action_published_at(action) is None:
                            skipped += 1
                            continue
                        actions[key] = action

                rows = [
                    row
                    for action in actions.values()
                    if (row := to_row(bank_id, action)) is not None
                ]

                inserted_count = db.upsert_raw_items(conn, rows)

                seen += len(rows)
                inserted += inserted_count

                print(
                    f"{bank_id}: {len(rows)} OCC actions seen, "
                    f"{inserted_count} inserted"
                    + (f", {skipped} skipped (no key/date)" if skipped else "")
                )

            except Exception as exc:
                try:
                    conn.rollback()
                except Exception:
                    pass

                failed.append(bank_id)
                print(f"{bank_id}: FAILED: {exc}", file=sys.stderr)

        db.write_heartbeat(
            conn,
            "poll_occ_enforcement",
            seen,
            inserted,
            time.monotonic() - started,
            not failed,
        )

    except Exception:
        try:
            conn.rollback()
            db.write_heartbeat(
                conn,
                "poll_occ_enforcement",
                seen,
                inserted,
                time.monotonic() - started,
                False,
            )
        except Exception:
            pass
        raise
    finally:
        conn.close()

    if failed:
        sys.exit(
            "poll_occ_enforcement: failed banks: "
            + ", ".join(failed)
        )


if __name__ == "__main__":
    sys.exit(main())