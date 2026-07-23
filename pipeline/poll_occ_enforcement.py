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

SEARCH_URL = "https://apps.occ.gov/EASearch/Search/Table"
BASE_URL = "https://apps.occ.gov"

# OCC目前默认分页10条，最稳定
PAGE_SIZE = 10
THROTTLE_S = 2.0


def clean(value):
    return re.sub(r"\s+", " ", value or "").replace("\xa0", " ").strip()


def parse_date(value):
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
            return datetime.strptime(value, fmt).replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            pass

    return None


def split_city_state(value):
    value = clean(value)
    if "," not in value:
        return value, ""

    city, state = value.rsplit(",", 1)
    return city.strip(), state.strip()


def parse_result_row(row):

    data = {}

    for td in row.select("td"):

        key = clean(td.get("data-th", ""))

        if not key:
            continue

        data[key] = clean(td.get_text(" ", strip=True))

        link = td.find("a", href=True)

        if link:
            data[key + "_url"] = urljoin(BASE_URL, link["href"])

    institution = data.get("Institution")

    if not institution:
        return None

    start_date = data.get("Start Date")

    city, state = split_city_state(data.get("City, State", ""))

    return {
        "institution": institution,
        "charter_number": data.get("Charter Number"),
        "company_name": data.get("Company Name"),
        "individual": data.get("Individual"),
        "city": city,
        "state": state,
        "action_type": data.get("Type"),
        "amount": data.get("Amount"),
        "start_date": start_date,
        "termination_date": data.get("Termination Date"),
        "docket_number": data.get("Docket Number"),
        "subject": data.get("Subject Matters"),
        "url": data.get("Start Doc_url"),
    }


def fetch_page(query, page):

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
        headers={
            "User-Agent": "Mozilla/5.0"
        },
    )

    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    page_text = clean(
        soup.get_text(" ", strip=True)
    )

    m = re.search(
        r"of\s+([\d,]+)\s+total\s+records",
        page_text,
        re.I,
    )

    total = int(m.group(1).replace(",", "")) if m else None

    actions = []

    for row in soup.select("tr.show-results"):

        action = parse_result_row(row)

        if action:
            actions.append(action)

    return actions, total
def fetch_all(query):

    results = []

    page = 0

    total = None

    while True:

        page_results, page_total = fetch_page(query, page)

        if total is None:
            total = page_total

        if not page_results:
            break

        results.extend(page_results)

        if total is not None and len(results) >= total:
            break

        page += 1

    return results


def external_id(action):

    if action["docket_number"]:
        return action["docket_number"]

    if action["url"]:
        return action["url"]

    return "|".join(
        filter(
            None,
            [
                action["institution"],
                action["action_type"],
                action["start_date"],
                action["individual"],
            ],
        )
    )


def to_row(bank_id, action):

    return {
        "source": "occ_enforcement",
        "external_id": external_id(action),
        "bank_id": bank_id,
        "published_at": parse_date(action["start_date"]),
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
            k: v
            for k, v in action.items()
            if k not in {"institution", "url"}
        },
    }


def bank_queries(bank):

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


def main():

    started = time.monotonic()

    seen = 0

    inserted = 0

    failed = []

    conn = db.connect()

    try:

        for bank in db.get_live_banks(conn):

            bank_id = bank["bank_id"]

            try:

                actions = {}

                for query in bank_queries(bank):

                    print(f"Searching OCC: {query}")

                    for action in fetch_all(query):

                        actions[external_id(action)] = action

                rows = [
                    to_row(bank_id, a)
                    for a in actions.values()
                ]

                n = db.upsert_raw_items(conn, rows)

                seen += len(rows)

                inserted += n

                print(
                    f"{bank_id}: "
                    f"{len(rows)} OCC actions seen, "
                    f"{n} inserted"
                )

            except Exception as exc:

                conn.rollback()

                failed.append(bank_id)

                print(
                    f"{bank_id}: FAILED: {exc}",
                    file=sys.stderr,
                )

        db.write_heartbeat(
            conn,
            "poll_occ_enforcement",
            seen,
            inserted,
            time.monotonic() - started,
            not failed,
        )

    finally:

        conn.close()

    if failed:

        sys.exit(
            "poll_occ_enforcement: failed banks: "
            + ", ".join(failed)
        )


if __name__ == "__main__":
    main()