"""Stage-1 quality gate: human gold slices vs Llama labels (scoring/DESIGN.md).

Joins evals/items/gold_slice_*.csv (human labels, id = raw_item_id) to the
labels_<date>.csv Kaggle round-trip output and reports agreement: overall,
per-class in both directions (human-side recall and llama-side), stratified
by slice, by source, and by the tone≠direction subgroup the labeling guide
flags as highest-value.

Slices 1-5 are a random sample. Slice 6 (export_new_gold_slice.py) was
deliberately stratified toward directional Llama labels, so the headline
gate number uses the random slices only; slice 6 strengthens the per-class
cells. Pooling everything would drag the overall number down by sampling
design, not labeler quality.

Run:
  python -m pipeline.quality_gate --gold-dir evals/items \
    --labels labels_2026-07-22.csv --output evals/gate_report_<date>.md
"""

import argparse
import csv
import glob
import os
import re
import sys
from collections import Counter

from pipeline.labeling import LABELS, validate_label

# Placeholder thresholds from DESIGN.md — TBD by team, shown in the report
# as provisional, never as a final per-class verdict.
PLACEHOLDER_OVERALL = 0.85
PLACEHOLDER_PER_CLASS = 0.75

# Acceptance criteria for a re-labeling run (prompt v3 onward), measured
# 2026-08-07 against prompt v2 and fixed BEFORE the run so they cannot be
# renegotiated after seeing the result.
#
# Every directional criterion is PAIRED. Precision alone is gameable: a
# labeler that almost never says `negative` scores high precision on the few
# it does say while missing every real one. Recall alone is gamed by saying
# `negative` to everything. Kappa is the primary because it resists both, and
# raw agreement is deliberately absent — on this corpus an all-`neutral`
# labeler beats the old 85% threshold (see chance_corrected).
CRITERIA = (
    ("kappa (primary)", ("chance", "kappa"), 0.60, "≥"),
    ("negative precision", ("llama_side", "negative"), 0.60, "≥"),
    ("negative recall — guards the above", ("human_side", "negative"), 0.85, "≥"),
    ("positive precision", ("llama_side", "positive"), 0.60, "≥"),
    ("neutral recall — guards over-correction", ("human_side", "neutral"), 0.82, "≥"),
)

# Slices sampled non-randomly (directional oversample); excluded from the
# headline number and reported as their own stratum.
STRATIFIED_SLICES = frozenset({"gold_slice_6"})

# The dev / holdout split of the human rows, fixed 2026-08-07 before any
# prompt revision. `dev` is what a prompt may be tuned against; `holdout` is
# read once, at the end, and is excluded from the FinBERT training set so
# there is a human-truth evaluation the model never trained on.
#
# Slice 6 is halved rather than assigned whole: it is the directional
# oversample, and giving it entirely to dev left the holdout with 3 negative
# rows. The cut is by file order, so it cannot be re-drawn in a favourable
# direction later, and the file is already shuffled (negative 4 vs 3, positive
# 10 vs 10 across the halves).
HOLDOUT_SLICES = frozenset({"gold_slice_2", "gold_slice_4", "gold_slice_5"})
HALVED_SLICE = "gold_slice_6"
HALVED_AT = 25


def stratum_for(slice_name: str, index: int) -> str:
    """Which side of the dev/holdout split one gold row falls on."""
    if slice_name == HALVED_SLICE:
        return "holdout" if index >= HALVED_AT else "dev"
    return "holdout" if slice_name in HOLDOUT_SLICES else "dev"


# ponytail: title heuristic from labeling_guide.md's trap tables — flagged
# rows are listed in the report for eyeball review; the flag is not a verdict.
TONE_DIRECTION_PATTERNS = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"strategic alternatives",
        r"consent order",
        r"upgrad|downgrad|price target|rating",
        # analyst actions that never say "downgrade" outright
        r"underweight|overweight|outperform|underperform|initiated coverage",
        r"short interest|put option|eps (forecast|estimate)|forecast cut",
        r"52[- ]week (high|low)",
        r"(purchase|buy|acquire)s? [\d,.]+ shares",
        r"stake in",
        r"dividend",
    )
)


def flag_tone_direction(title: str | None) -> bool:
    """Is this a title where financial tone and risk direction may diverge?"""
    return bool(title) and any(p.search(title) for p in TONE_DIRECTION_PATTERNS)


def chance_corrected(rows: list[dict]) -> dict:
    """Raw agreement is not readable on its own when one class dominates.

    A labeler that answers `neutral` to everything scores `majority` here, so
    a threshold set below that number carries no information. Cohen's kappa
    rescales agreement against what the two label distributions would produce
    by chance; macro-F1 weights the rare directional classes equally with
    neutral, which is what the distress index actually cares about.
    """
    n = len(rows)
    if not n:
        return {"n": 0}
    human = Counter(r["label"] for r in rows)
    llama = Counter(r["llama"] for r in rows)
    po = sum(r["agree"] for r in rows) / n
    pe = sum(human[c] * llama[c] for c in LABELS) / (n * n)
    f1s = []
    for c in LABELS:
        tp = sum(1 for r in rows if r["llama"] == r["label"] == c)
        prec = tp / max(1, llama[c])
        rec = tp / max(1, human[c])
        f1s.append(2 * prec * rec / max(1e-9, prec + rec))
    return {
        "n": n,
        "agreement": po,
        "majority": max(human.values()) / n,
        "kappa": (po - pe) / (1 - pe) if pe < 1 else 0.0,
        "macro_f1": sum(f1s) / len(LABELS),
    }


def compute_gate(gold_rows: list[dict], llama_by_id: dict[str, str]) -> dict:
    """Agreement stats over joined gold/llama rows. Pure, no I/O.

    gold_rows carry id, source, title, label, slice. All-or-nothing on the
    join: a gold id without a llama label raises (as import_labels does).
    """
    joined = []
    for r in gold_rows:
        if r["id"] not in llama_by_id:
            raise ValueError(f"no llama label for gold id {r['id']}")
        llama = llama_by_id[r["id"]]
        joined.append({**r, "llama": llama, "agree": llama == r["label"]})

    def tally(rows: list[dict]) -> dict:
        return {"n": len(rows), "agree": sum(r["agree"] for r in rows)}

    headline_rows = [r for r in joined if r["slice"] not in STRATIFIED_SLICES]
    confusion = Counter((r["llama"], r["label"]) for r in joined)
    per_class = {
        c: {
            "diag": confusion[(c, c)],
            "human_n": sum(n for (_, h), n in confusion.items() if h == c),
            "llama_n": sum(n for (m, _), n in confusion.items() if m == c),
        }
        for c in LABELS
    }
    flagged = [r for r in joined if flag_tone_direction(r["title"])]
    over_called = [
        r for r in joined if r["llama"] == "negative" and r["label"] == "neutral"
    ]
    return {
        "overall": tally(joined),
        "headline": tally(headline_rows),
        "chance": {
            "headline": chance_corrected(headline_rows),
            "pooled": chance_corrected(joined),
        },
        "confusion": dict(confusion),
        "per_class": per_class,
        "negative_over_call": {
            "rows": over_called,
            "rating_pattern": sum(
                1 for r in over_called if flag_tone_direction(r["title"])
            ),
        },
        "by_slice": {
            s: tally([r for r in joined if r["slice"] == s])
            for s in sorted({r["slice"] for r in joined})
        },
        "by_source": {
            s: tally([r for r in joined if r["source"] == s])
            for s in sorted({r["source"] for r in joined})
        },
        "by_stratum": {
            s: {
                **tally([r for r in joined if r.get("stratum") == s]),
                **chance_corrected([r for r in joined if r.get("stratum") == s]),
            }
            for s in ("dev", "holdout")
        },
        "tone_direction": {**tally(flagged), "rows": flagged},
        "disagreements": [r for r in joined if not r["agree"]],
    }


def evaluate_criteria(gate: dict) -> list[dict]:
    """Score a gate result against the pre-registered CRITERIA.

    Reported pooled (all slices) — the directional cells are what these
    criteria are about, and slice 6 is where most directional rows live.
    """
    out = []
    for name, (family, key), target, _op in CRITERIA:
        if family == "chance":
            actual = gate["chance"]["pooled"][key]
            n = gate["chance"]["pooled"]["n"]
        else:
            pc = gate["per_class"][key]
            n = pc["llama_n"] if family == "llama_side" else pc["human_n"]
            actual = pc["diag"] / n if n else 0.0
        out.append(
            {
                "name": name,
                "actual": actual,
                "target": target,
                "n": n,
                "passed": actual >= target,
            }
        )
    return out


def _pct(agree: int, n: int) -> str:
    return f"{agree}/{n} = {agree / n:.1%}" if n else "0/0 = n/a"


def _row_table(rows: list[dict]) -> list[str]:
    out = [
        "| id | slice | source | human | llama | title |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        out.append(
            f"| {r['id']} | {r['slice'].removeprefix('gold_slice_')} | {r['source']} "
            f"| {r['label']} | {r['llama']} | {r['title']} |"
        )
    return out


def render_report(gate: dict, run_date: str, labels_path: str) -> str:
    head, over = gate["headline"], gate["overall"]
    head_pass = head["agree"] / head["n"] >= PLACEHOLDER_OVERALL
    lines = [
        f"# Quality gate — {run_date}",
        "",
        f"Human labels: `evals/items/gold_slice_*.csv` ({over['n']} rows). "
        f"Llama labels: `{labels_path}` (`label_source='llama_kaggle'`, prompt v2).",
        "",
        "## Headline (random sample, slices 1–5)",
        "",
        f"Overall agreement **{_pct(head['agree'], head['n'])}** — placeholder "
        f"threshold ≥{PLACEHOLDER_OVERALL:.0%}: **{'PASS' if head_pass else 'FAIL'}** "
        "on the letter of the threshold. **Read the next section before acting "
        "on that word** — the threshold sits below the do-nothing baseline, so "
        "passing it carries no information.",
        "",
        f"Slice 6 is a deliberate directional oversample (20 neg / 20 pos / "
        f"10 neu on Llama's label, own-bank titles only — see "
        f"`export_new_gold_slice.py`), so it is excluded here; pooled over all "
        f"{over['n']} rows the number would read {_pct(over['agree'], over['n'])}, "
        "biased by sampling design, not labeler quality.",
        "",
        "## Chance-corrected reading (the number that matters)",
        "",
        "| | random (1–5) | pooled (1–6) |",
        "|---|---|---|",
        *(
            f"| {label} | {ch['headline'][key]:.3f} | {ch['pooled'][key]:.3f} |"
            for ch in [gate["chance"]]
            for label, key in (
                ("raw agreement", "agreement"),
                ("all-neutral baseline", "majority"),
                ("**Cohen's kappa**", "kappa"),
                ("macro F1", "macro_f1"),
            )
        ),
        "",
        f"**The gate threshold is set below the trivial baseline.** On the random "
        f"sample, {gate['chance']['headline']['majority']:.1%} of human labels are "
        f"`neutral`, so a labeler that answers `neutral` to every row scores "
        f"{gate['chance']['headline']['majority']:.1%} — above both the ≥"
        f"{PLACEHOLDER_OVERALL:.0%} threshold and Llama's own "
        f"{gate['chance']['headline']['agreement']:.1%}. Raw agreement cannot "
        "distinguish a working labeler from a broken one on this corpus; the "
        "team's threshold decision should be restated in kappa or macro-F1.",
        "",
        f"Kappa {gate['chance']['headline']['kappa']:.3f} (random) / "
        f"{gate['chance']['pooled']['kappa']:.3f} (pooled) is *moderate* agreement "
        "on the conventional reading — not a failure, not a pass.",
        "",
        "**Methodology limit worth stating in the report.** The human side is a "
        "second-pass *review* of slices 1–5, not a blind independent re-label, so "
        "the labels are better than a single pass but no inter-annotator "
        "agreement statistic exists — and a review cannot produce one, because a "
        "reviewer who sees the first answer mostly agrees with it. The ceiling "
        "these kappa numbers are measured against is therefore unknown.",
        "",
        "## Acceptance criteria for the next labeling run",
        "",
        "Fixed before the run so they cannot be renegotiated afterwards. Every "
        "directional criterion is **paired**: precision alone is passed by a "
        "labeler that stops saying the class at all, recall alone by one that "
        "says it everywhere. Raw agreement is deliberately not a criterion.",
        "",
        "| criterion | target | current (v2) | n | status |",
        "|---|---|---|---|---|",
        *(
            f"| {c['name']} | ≥{c['target']:.2f} | {c['actual']:.3f} | {c['n']} "
            f"| {'meets' if c['passed'] else '**below**'} |"
            for c in evaluate_criteria(gate)
        ),
        "",
        "Two of these demand improvement (kappa and negative precision — the "
        "defects v3 exists to fix); the rest sit at or just under today's value "
        "as **no-regression floors**, so a run that fixes negative by breaking "
        "positive or neutral fails. A floor already marked `meets` is not "
        "slack — it is the level that must survive.",
        "",
        "Read these as the bar the *next* run has to clear, not as a verdict on "
        "v2 — v2 is the measurement that set them. Note the sample sizes: on the "
        "directional rows one row moves the number by several points, so these "
        "detect a large effect, not a small one.",
        "",
        "## Per-class agreement (all rows, both directions)",
        "",
        "Placeholder threshold ≥75% per class — **the denominator direction is "
        "undefined in DESIGN.md, so both are reported and the per-class verdict "
        "is deferred to the team's threshold decision.**",
        "",
        "| class | human-side (recall) | llama-side (precision) |",
        "|---|---|---|",
    ]
    for c in LABELS:
        pc = gate["per_class"][c]
        lines.append(
            f"| {c} | {_pct(pc['diag'], pc['human_n'])} "
            f"| {_pct(pc['diag'], pc['llama_n'])} |"
        )
    neg = gate["per_class"]["negative"]
    oc = gate["negative_over_call"]
    lines += [
        "",
        f"**Key finding:** llama-side negative precision is "
        f"{_pct(neg['diag'], neg['llama_n'])} — when Llama says `negative`, "
        "humans usually see `neutral`. Llama over-calls negative; that is the "
        "systematic-bias pattern DESIGN.md asks this review to find.",
        "",
        "This is the finding that gates Stage 2, not the headline number. "
        "Directional labels are the whole training signal, and the training set "
        "carries only a few hundred `negative` rows to begin with — if most of "
        "them are `neutral` in truth, the fine-tune learns the error.",
        "",
        f"### Negative over-call ({len(oc['rows'])} rows: llama `negative`, "
        "human `neutral`)",
        "",
        f"{oc['rating_pattern']} of the {len(oc['rows'])} match the analyst/rating "
        "title heuristic — the bank is the *analyst* and another company is the "
        "subject. Those rows are the `keep for now` class in DESIGN.md's "
        "training-set hygiene table, so **the training-set filters do not remove "
        "them**: this is a labeling-prompt problem, not a filter problem. The "
        "remainder are ordinary news where the tone is negative but the bank's "
        "risk is not (`labeling_guide.md` trap tables).",
        "",
        *_row_table(oc["rows"]),
        "",
        "## Confusion (llama → human)",
        "",
        "| llama \\ human | " + " | ".join(LABELS) + " |",
        "|---|" + "---|" * len(LABELS),
    ]
    for m in LABELS:
        cells = " | ".join(str(gate["confusion"].get((m, h), 0)) for h in LABELS)
        lines.append(f"| {m} | {cells} |")
    lines += [
        "",
        "## Dev vs holdout",
        "",
        "`dev` is what a prompt may be tuned against; `holdout` is read once, at "
        "the end, and is the only human-truth evaluation the FinBERT training "
        "set is kept away from. A run that improves on dev while holdout drops "
        "is overfitted to the rows the prompt was written against.",
        "",
        "| stratum | n | agreement | kappa | macro F1 |",
        "|---|---|---|---|---|",
        *(
            f"| {s} | {t['n']} | {_pct(t['agree'], t['n'])} | {t['kappa']:.3f} "
            f"| {t['macro_f1']:.3f} |"
            for s, t in gate["by_stratum"].items()
            if t["n"]
        ),
        "",
        "⚠️ The holdout is **not fully blind**: this report lists every "
        "disagreeing row, holdout rows included, so anyone who read an earlier "
        "revision has seen them. Prompt revisions must be written from the dev "
        "rows and `labeling_guide.md` only, and the final write-up should call "
        "this holdout semi-blind rather than blind.",
        "",
        "## By slice",
        "",
        "| slice | agreement |",
        "|---|---|",
    ]
    for s, t in gate["by_slice"].items():
        note = " (stratified — see headline note)" if s in STRATIFIED_SLICES else ""
        lines.append(f"| {s} | {_pct(t['agree'], t['n'])}{note} |")
    lines += ["", "## By source", "", "| source | agreement |", "|---|---|"]
    for s, t in gate["by_source"].items():
        lines.append(f"| {s} | {_pct(t['agree'], t['n'])} |")
    td = gate["tone_direction"]
    lines += [
        "",
        "## Tone≠direction subgroup (title heuristic)",
        "",
        f"Agreement {_pct(td['agree'], td['n'])}. DESIGN.md requires reading the "
        "gate stratified on this subgroup — the heuristic flags candidate rows; "
        "eyeball them:",
        "",
        *_row_table(td["rows"]),
        "",
        f"## Disagreements ({len(gate['disagreements'])} rows)",
        "",
        *_row_table(gate["disagreements"]),
        "",
    ]
    return "\n".join(lines)


def load_gold_rows(gold_dir: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(gold_dir, "gold_slice_*.csv"))):
        slice_name = os.path.splitext(os.path.basename(path))[0]
        with open(path, encoding="utf-8") as f:
            for i, r in enumerate(csv.DictReader(f)):
                rows.append(
                    {
                        "id": r["id"],
                        "source": r["source"],
                        "title": r["title"],
                        "label": validate_label(r["label"]),
                        "slice": slice_name,
                        "stratum": stratum_for(slice_name, i),
                    }
                )
    return rows


def load_llama_labels(path: str) -> dict[str, str]:
    with open(path, encoding="utf-8") as f:
        return {r["raw_item_id"]: r["label"] for r in csv.DictReader(f)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold-dir", default="evals/items")
    ap.add_argument("--labels", required=True, help="labels_<date>.csv")
    ap.add_argument("--output", required=True, help="evals/gate_report_<date>.md")
    ap.add_argument("--run-date", required=True, help="YYYY-MM-DD")
    args = ap.parse_args()

    gate = compute_gate(load_gold_rows(args.gold_dir), load_llama_labels(args.labels))
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(render_report(gate, args.run_date, args.labels))
    head = gate["headline"]
    print(
        f"headline {_pct(head['agree'], head['n'])}, "
        f"{len(gate['disagreements'])} disagreements -> {args.output}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
