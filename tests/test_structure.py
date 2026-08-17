"""Design-conformance checks: the repo's structural rules, enforced by CI.

Each test names the rule it guards. If one fails, fix the change — not the
test — unless the team has explicitly agreed to change the rule.
"""

import hashlib
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def tracked(pattern):
    out = subprocess.run(
        ["git", "ls-files", pattern],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [ROOT / line for line in out.splitlines() if line]


def test_http_goes_through_the_shared_helper():
    """Rule: no poller/loader forks its own retry loop (RUNBOOK §6)."""
    offenders = [
        str(p.relative_to(ROOT))
        for p in tracked("*.py")
        if p.name != "http.py"
        and not str(p).startswith(str(ROOT / "tests"))
        and re.search(r"requests\.(get|post|request)\(", p.read_text())
    ]
    assert not offenders, f"use pipeline.http.throttled_get instead: {offenders}"


def test_executable_code_stays_in_owned_dirs():
    """Rule: no executable Python outside the dirs that own code —
    db/, pipeline/, eda/, evals/, index/, tests/, dashboard/, and Shu Han's dataset
    module."""
    allowed = (
        "db/",
        "pipeline/",
        "eda/",
        "evals/",
        "index/",
        "tests/", "dashboard/",
        "unified_ffiec_fdic_dataset/",
    )
    offenders = [
        str(p.relative_to(ROOT))
        for p in tracked("*.py")
        if p.stat().st_size > 0 and not str(p.relative_to(ROOT)).startswith(allowed)
    ]
    assert not offenders, f"executable code outside owned dirs: {offenders}"


def test_applied_migrations_are_immutable():
    """Rule: never edit an applied migration — add a new numbered one.

    Adding a migration: append its sha256 to db/migrations/CHECKSUMS
    (shasum -a 256 00X_*.sql) in the same commit.
    """
    recorded = {}
    for line in (ROOT / "db/migrations/CHECKSUMS").read_text().splitlines():
        digest, name = line.split()
        recorded[name] = digest
    actual = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (ROOT / "db/migrations").glob("0*.sql")
    }
    changed = [n for n in recorded if recorded[n] != actual.get(n)]
    assert not changed, f"applied migrations were edited: {changed}"
    unrecorded = sorted(set(actual) - set(recorded))
    assert not unrecorded, f"new migrations missing from CHECKSUMS: {unrecorded}"


def test_workflow_matrix_modules_exist():
    """Rule: every matrix entry is a runnable pipeline module (and vice versa
    catches typos before they become silently-skipped sources)."""
    yml = (ROOT / ".github/workflows/ingest.yml").read_text()
    m = re.search(r"module:\s*\[([^\]]*)\]", yml)
    assert m, "poll matrix not found in ingest.yml"
    for module in (x.strip() for x in m.group(1).split(",")):
        assert (ROOT / "pipeline" / f"{module}.py").exists(), (
            f"matrix entry '{module}' has no pipeline/{module}.py"
        )


def test_raw_item_sources_match_check_constraint():
    """Rule: every source a poller writes must be allowed by the newest
    raw_item CHECK constraint."""
    latest_check = None
    for sql in sorted((ROOT / "db/migrations").glob("0*.sql")):
        m = re.search(r"CHECK \(source IN \(([^)]*)\)\)", sql.read_text())
        if m:
            latest_check = {s.strip().strip("'") for s in m.group(1).split(",")}
    assert latest_check, "no raw_item source CHECK found"
    for p in tracked("pipeline/poll_*.py"):
        m = re.search(r'"source":\s*"([a-z_]+)"', p.read_text())
        if m:
            assert m.group(1) in latest_check, (
                f"{p.name} writes source '{m.group(1)}' "
                f"not in CHECK {sorted(latest_check)}"
            )
