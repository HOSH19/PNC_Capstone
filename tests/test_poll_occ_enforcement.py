"""Contract tests for OCC enforcement poller name matching."""

from pipeline import poll_occ_enforcement as occ
from pipeline.poll_fdic_enforcement import build_matcher

BANKS = [
    {
        "bank_id": "pnc",
        "bank_legal_name": "PNC Bank, National Association",
        "holding_name": "PNC Financial Services Group, Inc.",
        "aliases": ["PNC", "PNC Bank"],
    },
    {
        "bank_id": "wfc",
        "bank_legal_name": "Wells Fargo Bank, National Association",
        "holding_name": "Wells Fargo & Company",
        "aliases": ["Wells Fargo"],
    },
]


def test_search_hit_for_other_bank_is_skipped():
    match = build_matcher(BANKS)
    action = {"institution": "Wells Fargo Bank, National Association"}
    assert match(action["institution"]) == "wfc"
    assert match(action["institution"]) != "pnc"


def test_main_skips_misattributed_actions(monkeypatch, fake_db):
    fake_db.banks = BANKS
    calls = []

    def fake_fetch_all(query):
        calls.append(query)
        if query == "PNC Bank":
            return [
                {
                    "institution": "PNC Bank, National Association",
                    "charter_number": "6384",
                    "company_name": "",
                    "individual": "",
                    "city": "Pittsburgh",
                    "state": "PA",
                    "action_type": "Consent Order",
                    "amount": "",
                    "start_date": "01/15/2024",
                    "termination_date": "",
                    "docket_number": "AA-EC-2024-001",
                    "subject": "BSA",
                    "url": "https://apps.occ.gov/example/pnc.pdf",
                },
                {
                    "institution": "Wells Fargo Bank, National Association",
                    "charter_number": "3511",
                    "company_name": "",
                    "individual": "",
                    "city": "Sioux Falls",
                    "state": "SD",
                    "action_type": "Consent Order",
                    "amount": "",
                    "start_date": "02/01/2024",
                    "termination_date": "",
                    "docket_number": "AA-EC-2024-002",
                    "subject": "Risk",
                    "url": "https://apps.occ.gov/example/wfc.pdf",
                },
            ]
        return []

    monkeypatch.setattr(occ, "fetch_all", fake_fetch_all)

    occ.main()

    assert len(fake_db.rows) == 1
    row = fake_db.rows[0]
    assert row["bank_id"] == "pnc"
    assert row["external_id"] == "AA-EC-2024-001"
    assert fake_db.heartbeats == [("poll_occ_enforcement", 1, 1, True)]
