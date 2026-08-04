"""Pull a wide FFIEC Call Report panel from the CDR bulk archive.

Same source and library the team's existing `unified_ffiec_fdic_dataset` collector
uses — this one just keeps every field instead of the nine that build outputs.

The RCFD/RCON split is the whole difficulty. FFIEC 031 filers (banks with foreign
offices — that is, the large ones) report consolidated figures under RCFD and
leave the RCON column empty; FFIEC 041 filers report domestic-only under RCON and
have no RCFD. Same MDRM item, two prefixes, and no bank fills both consistently.

v2 got this wrong in a way that was invisible in aggregate: it applied the
coverage filter BEFORE stripping prefixes, so RCFD columns — present for only
~1.8% of banks — never cleared the 50% threshold and were dropped whole. The
banks that lost their data were exactly the large ones. Silicon Valley Bank,
First Republic and JPMorgan all came out with NaN total assets, and roughly a
third of the 104 tracked banks scored on imputed medians. Aggregate coverage
still read 98.2%, and the positive-vs-negative missingness check passed cleanly,
because the affected banks are few and nearly all survivors.

So here: coalesce first, filter second, and prefer the consolidated figure.
"""
import os
import re
import sys
import zipfile

import pandas as pd
from ffiec_data_collector import FFIECDownloader, FileFormat, Product

OUT = os.environ.get("FFIEC_OUT", "/tmp/ffiec_raw")
SCHEDULES = [
    "RC", "RI", "RCN", "RCO", "RCB", "RCCI", "RCE", "RCRI", "RCL", "RCD",
    # RI-B I/II give charge-offs, recoveries and the allowance roll-forward;
    # RC-K gives quarterly average assets, the correct ROA denominator; RC-R II
    # gives risk-weighted assets. All distress-relevant and all missing in v2.
    "RIBI", "RIBII", "RCK", "RCRII", "RCF", "RCG", "RCM",
]
# Deliberately permissive. Extraction should not be making modelling decisions:
# a consolidated-only item is populated for ~1.8% of banks and a 50% threshold
# deletes it outright, which is how the large banks lost their data twice. Keep
# anything a real number of banks report and let features.py screen it on both
# the label axis and the size axis.
MIN_COV = float(os.environ.get("FFIEC_MIN_COV", "0.015"))
START = os.environ.get("FFIEC_START", "20010101")

# Consolidated beats domestic-only beats foreign-only. RCFA/RCFW are the RC-R
# capital-schedule consolidated forms — omitted in the first pass, which left
# them ranked below the domestic RCOA column and inverted the priority for that
# whole schedule.
PREFIX_RANK = {
    "RCFD": 0, "RCFA": 0, "RCFW": 1, "RCFN": 2,
    "RCON": 3, "RCOA": 3, "RCOW": 4,
    "RIAD": 0, "RIBI": 0,
}


def schedule_files(z, code):
    """Every part of a schedule — RCB/RCO/RCL ship as '(1 of 2)' and so on."""
    pat = re.compile(rf"Schedule {code} \d{{8}}(\(\d of \d\))?\.txt$")
    return [n for n in z.namelist() if pat.search(n)]


def read_part(z, name):
    """One schedule file -> a frame indexed by IDRSSD, columns still prefixed."""
    txt = z.open(name).read().decode("utf-8", "replace")
    lines = [l for l in txt.split("\n") if l.strip()]
    if len(lines) < 3:
        return None
    hdr = [c.strip('"') for c in lines[0].split("\t")]
    rows = [l.split("\t") for l in lines[2:]]
    df = pd.DataFrame(rows, columns=hdr)
    if "IDRSSD" not in df.columns:
        return None
    df["IDRSSD"] = pd.to_numeric(df["IDRSSD"].str.strip('"'), errors="coerce")
    return df.dropna(subset=["IDRSSD"]).set_index("IDRSSD")


def parse_schedule(z, code):
    """Merge every part of a schedule, THEN coalesce RCFD/RCON by MDRM item.

    Doing this per file is wrong and was the second half of the same bug. FFIEC
    ships RC-B, RC-L, RC-N and RC-O as '(1 of 2)'/'(2 of 2)', and the split puts
    an item's consolidated column in one part and its domestic twin in the other
    — RCFDK213 sits in RC-N part 1 with 1.76% coverage while RCONK213 sits in
    part 2 with 98.24%, disjoint, union covering every filer. Coalescing within
    a file never sees the pair, so the consolidated half fails the coverage
    threshold alone and the large banks lose the field. On 2013Q1 that removed
    193 of 687 fields for the 95 largest filers, most of them RC-N past-due and
    nonaccrual lines.
    """
    parts = [p for p in (read_part(z, n) for n in schedule_files(z, code))
             if p is not None]
    if not parts:
        return None
    idx = parts[0].index
    for p in parts[1:]:
        idx = idx.union(p.index)

    by_item = {}
    for p in parts:
        p = p.reindex(idx)
        for c in p.columns:
            if len(c) < 5:
                continue
            by_item.setdefault(c[4:], []).append((c[:4], p[c]))

    keep = {}
    for item, cols in by_item.items():
        cols.sort(key=lambda t: PREFIX_RANK.get(t[0], 9))
        merged = None
        for _, s in cols:
            v = pd.to_numeric(s.astype(str).str.strip('"'), errors="coerce")
            merged = v if merged is None else merged.fillna(v)
        if merged.notna().mean() >= MIN_COV:
            keep[item] = merged
    if not keep:
        return None
    out = pd.DataFrame(keep)
    out.index = idx
    return out


def quarter_frame(zip_path, quarter):
    with zipfile.ZipFile(zip_path) as z:
        parts = []
        for code in SCHEDULES:
            p = parse_schedule(z, code)
            if p is not None:
                p.columns = [f"{code}_{c}" for c in p.columns]
                parts.append(p)
        if not parts:
            return None
        df = pd.concat(parts, axis=1)
        dup = df.columns.duplicated()
        if dup.any():
            print(f"    WARN {quarter}: {dup.sum()} 个重复列名", flush=True)
            df = df.loc[:, ~dup]
        df["quarter"] = quarter
        return df.rename_axis("IDRSSD").reset_index()


def main():
    os.makedirs(OUT, exist_ok=True)
    dl = FFIECDownloader()
    quarters = sorted(q for q in dl.get_bulk_data_sources_cdr()["available_quarters"]
                      if q >= START)
    print(f"{len(quarters)} quarters: {quarters[0]} ~ {quarters[-1]}", flush=True)
    ok = 0
    for i, q in enumerate(quarters, 1):
        dst = f"{OUT}/{q}.parquet"
        if os.path.exists(dst):
            ok += 1
            continue
        try:
            r = dl.download(Product.CALL_SINGLE, q, FileFormat.TSV)
            if not r.success:
                print(f"  [{i}/{len(quarters)}] {q}: DOWNLOAD FAILED", flush=True)
                continue
            df = quarter_frame(r.file_path, q)
            os.remove(r.file_path)              # keep disk flat
            if df is None:
                print(f"  [{i}/{len(quarters)}] {q}: NO PARSABLE SCHEDULES", flush=True)
                continue
            df.to_parquet(dst, index=False)
            ok += 1
            ta = df["RC_2170"] if "RC_2170" in df else pd.Series(dtype=float)
            print(f"  [{i}/{len(quarters)}] {q}: {len(df):,} banks x {df.shape[1]-2} fields"
                  f"  总资产缺失 {int(ta.isna().sum())}", flush=True)
        except Exception as e:
            print(f"  [{i}/{len(quarters)}] {q}: ERROR {type(e).__name__}: {e}", flush=True)
    print(f"\ndone: {ok}/{len(quarters)} quarters -> {OUT}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
