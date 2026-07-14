"""Contract tests for the Fed enforcement CSV poller."""

import types

import pytest

from pipeline import poll_fed_enforcement as pfe
from pipeline.poll_fdic_enforcement import build_matcher

BANKS = [
    {"bank_id": "wfc", "bank_legal_name": "Wells Fargo Bank, National Association",
     "holding_name": "Wells Fargo & Company", "aliases": ["Wells Fargo"]},
    {"bank_id": "pnc", "bank_legal_name": "PNC Bank, National Association",
     "holding_name": "PNC Financial Services Group, Inc.", "aliases": ["PNC", "PNC Bank"]},
]

CSV_HEADER = ("Effective Date,Termination Date,Individual,Individual Affiliation,"
              "Banking Organization,Action,URL,Name,Note")


def test_location_suffixes_and_joined_orgs_are_matched():
    match = build_matcher(BANKS)
    hits = pfe.match_organizations(
        '"Wells Fargo & Company, San Francisco, California and '
        'Some Tiny Bancorp, Inc., Nowhere, Kansas"'.strip('"'), match)
    assert hits == {"wfc"}
    assert pfe.match_organizations("Unrelated Bancorp, Omaha, Nebraska", match) == set()


def wire(monkeypatch, fake_db, csv_body):
    fake_db.banks = BANKS
    resp = types.SimpleNamespace(content=csv_body.encode())
    monkeypatch.setattr(pfe, "throttled_get", lambda url, **kw: resp)


def test_end_to_end_inserts_matched_actions(monkeypatch, fake_db, capsys):
    wire(monkeypatch, fake_db, CSV_HEADER + "\n"
         '2018-02-02,,,,"Wells Fargo & Company, San Francisco, California",'
         "Consent Order,https://www.federalreserve.gov/x.htm,Press Release,\n"
         '2020-01-15,,John Doe,Former CEO,,Prohibition,https://frb.gov/y.htm,,\n'
         '2019-06-01,,,,"Small Town Bancorp, Elsewhere, Iowa",Written Agreement,,,\n')
    pfe.main()
    assert len(fake_db.rows) == 1
    row = fake_db.rows[0]
    assert (row["source"], row["bank_id"]) == ("fed_enforcement", "wfc")
    assert row["published_at"].year == 2018
    assert row["meta"]["action"] == "Consent Order"
    assert fake_db.heartbeats == [("poll_fed_enforcement", 1, 1, True)]
    assert "2 skipped" in capsys.readouterr().out


def test_rerun_is_idempotent(monkeypatch, fake_db):
    body = (CSV_HEADER + "\n"
            '2018-02-02,,,,"Wells Fargo & Company, San Francisco, California",'
            "Consent Order,https://www.federalreserve.gov/x.htm,,\n")
    wire(monkeypatch, fake_db, body)
    pfe.main()
    pfe.main()
    assert len(fake_db.rows) == 1
    assert fake_db.heartbeats[1][2] == 0


def test_schema_drift_fails_loudly(monkeypatch, fake_db):
    wire(monkeypatch, fake_db, "totally,different,columns\n1,2,3\n")
    with pytest.raises(RuntimeError, match="missing columns"):
        pfe.main()
