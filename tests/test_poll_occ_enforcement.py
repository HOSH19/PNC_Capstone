"""Contract tests for OCC enforcement poller."""

from bs4 import BeautifulSoup

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

SAMPLE_ROW = """
<tr class="show-results">
  <td data-th="Institution">PNC Bank, National Association</td>
  <td data-th="Charter Number">6384</td>
  <td data-th="City, State">Pittsburgh, PA</td>
  <td data-th="Type">Consent Order</td>
  <td data-th="Start Date">01/15/2024</td>
  <td data-th="Docket Number">AA-EC-2024-001</td>
  <td data-th="Subject Matters">BSA compliance</td>
  <td data-th="Start Doc"><a href="/EASearch/GetFile?id=1">PDF</a></td>
</tr>
"""


def test_parse_result_row_from_fixture():
    row = BeautifulSoup(SAMPLE_ROW, "html.parser").select_one("tr.show-results")
    action = occ.parse_result_row(row)
    assert action is not None
    assert action["institution"] == "PNC Bank, National Association"
    assert action["docket_number"] == "AA-EC-2024-001"
    assert action["city"] == "Pittsburgh"
    assert action["state"] == "PA"
    assert action["url"].endswith("/EASearch/GetFile?id=1")


def test_external_id_disambiguates_shared_docket():
    base = {
        "docket_number": "AA-EC-2024-001",
        "institution": "PNC Bank, National Association",
        "individual": "",
        "url": "",
        "action_type": "Consent Order",
        "start_date": "01/15/2024",
    }
    other = {**base, "institution": "Wells Fargo Bank, National Association"}
    assert occ.external_id(base) != occ.external_id(other)


def test_to_row_skips_unparseable_date():
    action = {
        "institution": "PNC Bank, National Association",
        "charter_number": "6384",
        "company_name": "",
        "individual": "",
        "city": "Pittsburgh",
        "state": "PA",
        "action_type": "Consent Order",
        "amount": "",
        "start_date": "N/A",
        "termination_date": "",
        "docket_number": "AA-EC-2024-001",
        "subject": "BSA",
        "url": "https://apps.occ.gov/example/pnc.pdf",
    }
    assert occ.to_row("pnc", action) is None


def test_parse_fixture_to_row():
    row = BeautifulSoup(SAMPLE_ROW, "html.parser").select_one("tr.show-results")
    action = occ.parse_result_row(row)
    out = occ.to_row("pnc", action)
    assert out["bank_id"] == "pnc"
    assert out["external_id"].startswith("AA-EC-2024-001|")
    assert out["published_at"].year == 2024


def test_search_hit_for_other_bank_is_skipped():
    match = build_matcher(BANKS)
    action = {"institution": "Wells Fargo Bank, National Association"}
    assert match(action["institution"]) == "wfc"
    assert match(action["institution"]) != "pnc"


def test_main_skips_misattributed_actions(monkeypatch, fake_db):
    fake_db.banks = BANKS

    def fake_fetch_all(query):
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
    assert row["external_id"].startswith("AA-EC-2024-001|")
    assert fake_db.heartbeats == [("poll_occ_enforcement", 1, 1, True)]
