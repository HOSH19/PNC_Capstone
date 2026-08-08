"""Build the MDRM item -> description map that features.py's type screen needs.

Screen [4] drops identifiers, dates and near-constants. The near-constant half is
statistical and needs nothing external, but the identifier half cannot be: RI_9106
is an ACQUISITION DATE and RCO_A545 is an FDIC CERTIFICATE NUMBER, and once divided
by total assets both pass every coverage and variance test with AUC 0.501. The only
thing that distinguishes them from a real quantity is what the field is called.

The names live in row 2 of every schedule file in the same CDR archive extract.py
already downloads -- row 1 is the MDRM codes, row 2 the descriptions, data from row
3 on. Nothing external is required.

This ran as an ad-hoc step during development and wrote to /tmp, which is why the
output went missing and screen [4] silently degraded: `names = {}` drops nobody, the
candidate pool came out 534 instead of 529, and no error was raised. The output is
committed now, and features.py fails loudly if it is absent.

Run only when the field set changes -- a new schedule, or a rebuild reporting
missing descriptions:

    python3 index/fundamentals/mdrm_names.py
"""
import json
import os
import pathlib
import sys
import zipfile

from ffiec_data_collector import FFIECDownloader, FileFormat, Product

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from extract import SCHEDULES, schedule_files

OUT = os.environ.get("MDRM_NAMES",
                     str(pathlib.Path(__file__).resolve().parent / "mdrm_names.json"))
# Spread across the modelled window. Items enter and leave the Call Report, so one
# quarter does not cover the field set: the RC-R II risk-weighting lines arrive in
# 2015 and several RC-B fair-value lines are dropped after 2020.
QUARTERS = os.environ.get(
    "MDRM_QUARTERS",
    "20170331,20180331,20190331,20200331,20210331,20220331,20230331,20240331,"
    "20250331,20260331").split(",")


def log(m):
    print(m, flush=True)


def names_from_zip(path):
    """{schedule}_{item} -> description, read from row 2 of each schedule file."""
    out = {}
    with zipfile.ZipFile(path) as z:
        for code in SCHEDULES:
            for name in schedule_files(z, code):
                txt = z.open(name).read().decode("utf-8", "replace")
                lines = txt.split("\n")
                if len(lines) < 2:
                    continue
                cols = [c.strip('"') for c in lines[0].split("\t")]
                desc = [c.strip('"').strip() for c in lines[1].split("\t")]
                for col, d in zip(cols, desc):
                    # col is a 4-char reporting prefix plus the 4-char MDRM item;
                    # features.py keys on {schedule}_{item}, so the prefix goes.
                    if len(col) < 5 or not d:
                        continue
                    out.setdefault(f"{code}_{col[4:]}", d)
    return out


def main():
    dl = FFIECDownloader()
    merged = {}
    for i, q in enumerate(QUARTERS, 1):
        try:
            r = dl.download(Product.CALL_SINGLE, q, FileFormat.TSV)
            if not r.success:
                log(f"  [{i}/{len(QUARTERS)}] {q}: DOWNLOAD FAILED")
                continue
            got = names_from_zip(r.file_path)
            os.remove(r.file_path)
            new = len(set(got) - set(merged))
            # Last write wins, and QUARTERS runs oldest to newest, so a field
            # keeps its most recent label. 83 items have been relabelled since
            # 2017 -- ALLL became ACL under CECL, "troubled debt restructuring"
            # became "loan modification" under ASU 2022-02, "capitalized leases"
            # became "right-of-use" under ASC 842. First-wins would pin all of
            # them to their 2017 wording. A field dropped from the Call Report
            # keeps the last label it was filed under, which is what it should.
            merged.update(got)
            log(f"  [{i}/{len(QUARTERS)}] {q}: {len(got):,} fields, {new:,} new")
        except Exception as e:
            log(f"  [{i}/{len(QUARTERS)}] {q}: ERROR {type(e).__name__}: {e}")

    if not merged:
        log("no field descriptions retrieved -- not writing the file")
        return 1
    with open(OUT, "w") as f:
        json.dump(dict(sorted(merged.items())), f, indent=0, sort_keys=True)
    log(f"\n{len(merged):,} field descriptions -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
