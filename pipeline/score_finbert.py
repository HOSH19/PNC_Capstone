"""Stage 3 — score pending raw_item rows with the fine-tuned FinBERT.

Drains the `finbert_status = 'pending'` queue: eligible rows get a row in
`item_score`, ineligible ones are marked `skipped` without being scored.
Both writes happen in one transaction per batch, so a crash leaves the queue
consistent rather than double-scoring on the next run.

Eligibility comes from `pipeline.eligibility` — the same filter the labeling
export used. That is the point of it: if serving scored a different set than
training saw, the model would be applied to a distribution it never learned.

**probs is not decoration.** The model under-calls direction (it distills a
labeler whose `negative` recall is 0.857 on 14 gold rows, and it trained on 38
`negative` examples), so the aggregation layer is expected to lower the
directional threshold rather than take argmax. Storing the distribution is
what makes that possible without re-scoring — see scoring/DESIGN.md, "Training
on a champion that missed the criteria", obligation 2.

Weights live outside the repo (`chloejiwon/finbert-ft-2026-08-09`); pass the
downloaded directory with --model-dir. torch and transformers are not in
requirements for the same reason they are not for the Kaggle scripts: nothing
in CI imports this module unless it is being run.

    python -m pipeline.score_finbert --model-dir ./finbert_ft --limit 5000
"""

import argparse
import json
import sys
import time

from pipeline import db, eligibility
from pipeline.labeling import LABELS

# Rows per transaction. Small enough that a crash costs little re-work, large
# enough that the per-batch commit is not the bottleneck.
BATCH = 500
# calibration: GDELT rows are title-only and short; EDGAR excerpts are long.
# 256 matches what the fine-tune used — scoring at a different length would
# feed the model inputs it was not trained on.
MAX_LENGTH = 256


def triage(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split fetched rows into (scorable, skipped). Pure, no model, no DB.

    A row that `eligibility` rejects is not an error: it is a row serving
    should never score, and it gets `finbert_status='skipped'` with the
    reason recorded so the funnel stays auditable.
    """
    scorable, skipped = [], []
    for r in rows:
        try:
            result = eligibility.check(r)
        except eligibility.UnknownSource as exc:
            # A source with no adapter is a deploy mistake, not data: fail
            # loudly rather than silently skipping every row of it.
            raise RuntimeError(f"no eligibility adapter for row {r['id']}") from exc
        if result.eligible:
            scorable.append({**r, "text": result.text})
        else:
            skipped.append({**r, "reason": result.reason})
    return scorable, skipped


def to_score_row(raw_id: int, label: str, probs: dict, model_version: str) -> dict:
    return {
        "raw_item_id": raw_id,
        "label": label,
        "probs": json.dumps(probs),
        "model_version": model_version,
    }


def fetch_pending(conn, limit: int, newest_first: bool = False) -> list[dict]:
    """Oldest first by default; newest first for the incremental job.

    The queue is FIFO by id and the 2020-2024 backfill holds the low ids, so
    a CPU run draining oldest-first would chew historical rows forever and
    never reach today's. The incremental job wants the opposite end; the
    backlog job wants this end, and both leave the other's rows pending.
    """
    order = "DESC" if newest_first else "ASC"
    with conn.cursor() as cur:
        cur.execute(
            f"""SELECT id, source, title, text_excerpt, meta
                FROM raw_item
                WHERE finbert_status = 'pending'
                ORDER BY id {order}
                LIMIT %(limit)s""",
            {"limit": limit},
        )
        return cur.fetchall()


def write_batch(conn, scores: list[dict], skipped: list[dict]) -> None:
    """One transaction: the scores, then the queue flags that retire them."""
    with conn.cursor() as cur:
        if scores:
            cur.executemany(
                """INSERT INTO item_score
                       (raw_item_id, label, probs, model_version)
                   VALUES (%(raw_item_id)s, %(label)s, %(probs)s, %(model_version)s)
                   ON CONFLICT (raw_item_id) DO UPDATE SET
                       label = EXCLUDED.label,
                       probs = EXCLUDED.probs,
                       model_version = EXCLUDED.model_version,
                       scored_at = now()""",
                scores,
            )
            cur.execute(
                "UPDATE raw_item SET finbert_status = 'done' WHERE id = ANY(%s)",
                ([s["raw_item_id"] for s in scores],),
            )
        if skipped:
            cur.execute(
                "UPDATE raw_item SET finbert_status = 'skipped' WHERE id = ANY(%s)",
                ([r["id"] for r in skipped],),
            )
    conn.commit()


def load_model(model_dir: str):
    """Imported here, not at module scope: the pure functions above are
    tested in CI, where torch is deliberately absent."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    id2label = {int(i): lb for i, lb in model.config.id2label.items()}
    assert sorted(id2label.values()) == sorted(LABELS), id2label
    return torch, tokenizer, model, id2label


def predict(torch, tokenizer, model, id2label, texts: list[str]) -> list[tuple]:
    enc = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1)
    out = []
    for row in probs:
        dist = {id2label[i]: round(float(p), 4) for i, p in enumerate(row)}
        out.append((max(dist, key=dist.get), dist))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True, help="downloaded weights dir")
    ap.add_argument(
        "--model-version",
        default=None,
        help="defaults to metrics.json's, else the dir name",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=100_000,
        help="rows to drain this run; the queue survives restarts",
    )
    ap.add_argument(
        "--batch",
        type=int,
        default=BATCH,
        help="rows per transaction; raise it when the DB round "
        "trip dominates, as it does draining the backlog "
        "from Kaggle",
    )
    args = ap.parse_args()

    version = args.model_version
    if not version:
        try:
            with open(f"{args.model_dir}/metrics.json", encoding="utf-8") as f:
                version = json.load(f)["model_version"]
        except (OSError, KeyError):
            version = args.model_dir.rstrip("/").split("/")[-1]

    started = time.monotonic()
    torch, tokenizer, model, id2label = load_model(args.model_dir)
    scored = skipped_n = 0
    conn = db.connect()
    try:
        while scored + skipped_n < args.limit:
            take = min(args.batch, args.limit - scored - skipped_n)
            rows = fetch_pending(conn, take, args.newest_first)
            if not rows:
                break
            scorable, skipped = triage(rows)
            preds = (
                predict(
                    torch, tokenizer, model, id2label, [r["text"] for r in scorable]
                )
                if scorable
                else []
            )
            write_batch(
                conn,
                [
                    to_score_row(r["id"], lb, pr, version)
                    for r, (lb, pr) in zip(scorable, preds, strict=True)
                ],
                skipped,
            )
            scored += len(scorable)
            skipped_n += len(skipped)
            print(f"  {scored} scored, {skipped_n} skipped", file=sys.stderr)
        db.write_heartbeat(
            conn,
            "score_finbert",
            scored + skipped_n,
            scored,
            time.monotonic() - started,
            True,
        )
    except Exception:
        try:
            conn.rollback()
            db.write_heartbeat(
                conn,
                "score_finbert",
                scored + skipped_n,
                scored,
                time.monotonic() - started,
                False,
            )
        except Exception:
            pass  # never mask the original failure
        raise
    finally:
        conn.close()
    print(f"{version}: {scored} scored, {skipped_n} skipped", file=sys.stderr)


if __name__ == "__main__":
    main()
