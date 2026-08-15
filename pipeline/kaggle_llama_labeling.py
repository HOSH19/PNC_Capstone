"""Kaggle GPU driver: label the export CSV with Llama via vLLM.

This file is GPU-bound glue only — pure logic is in pipeline/labeling.py and
pipeline/eligibility.py, which it reuses. It is NOT part of the ingestion
pipeline and is never imported by it; it runs by hand in a Kaggle notebook.

**Use kaggle_llama_labeling.ipynb**, which carries out the procedure below and
the smoke test. The steps stay here because the notebook is a convenience
wrapper, not the definition.

Run procedure (Kaggle):
  1. New Kaggle notebook → Settings → Accelerator = GPU (T4 x2 or P100),
     Internet = On.
  2. Upload one PRIVATE Kaggle dataset holding the export CSV
     (labeling_batch_<date>.csv), the prompt, and the four modules this
     driver imports — pipeline/{__init__,labeling,eligibility}.py plus this
     file. Private because the batch CSVs are gitignored data.
  3. Copy the upload into /kaggle/working (read-only inputs cannot host the
     package) and chdir there so `pipeline.` resolves.
  4. pip install vllm  (first run downloads weights, a few minutes).
  5. python -m pipeline.kaggle_llama_labeling \
       --input labeling_batch_<date>.csv \
       --prompt prompts/jiwon_llama_v4.md \
       --output labels_<date>.csv --run-date <YYYY-MM-DD>
  6. Download labels_<date>.csv → pipeline.quality_gate to score it, then the
     repo import script.

Smoke test (required on first run):
  - 20-row mini CSV end-to-end: every row gets a valid label, schema is
    (raw_item_id, label, model_meta), model_meta is JSON.
  - Re-run with same input gives identical labels (temperature=0).
  - Class distribution is sane (not all neutral).

vLLM's constrained-decoding and chat APIs are VERSION-SENSITIVE, and every
version-dependent line is marked `calibration`. Constrained decoding is
already handled both ways in build_sampling_params; if `.chat()` fails on a
future version, it may need a manual chat-template apply on `.generate()`.
Confirm the AWQ checkpoint path is reachable from Kaggle too.
"""

import argparse
import csv
import json

from vllm import LLM, SamplingParams

from pipeline.eligibility import text_for
from pipeline.labeling import (
    LABELS,
    build_model_meta,
    parse_prompt_version,
    render_prompt,
    validate_label,
)

# calibration: confirm this AWQ checkpoint is reachable on Kaggle.
MODEL = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
QUANTIZATION = "awq"
# calibration: Llama 3.1 advertises a 131072-token context, and vLLM sizes the
# KV cache for it up front — 16 GiB, which a 16 GB T4 cannot give after the
# 5.4 GiB of weights (~7 GiB is left, and startup fails outright). Our prompts
# are nowhere near that: the longest rendered prompt is ~8,030 characters on
# prompt v3 and ~9,080 on the longer v4, so roughly 2,300-3,100 tokens
# depending on how the EDGAR digit-heavy boilerplate tokenizes. 6144 keeps a
# 2x margin over that while the KV reservation stays ~21x smaller than the
# native context — well inside the ~7 GiB available. Grow it whenever the
# prompt or a text field grows: vLLM rejects an over-long prompt rather than
# truncating it, so overflow costs the whole run, and KV memory is not the
# binding constraint at this size.
MAX_MODEL_LEN = 6144
# The prompt version is NOT a constant here — it is read from the prompt file
# passed to --prompt, so pointing this at a v4 file cannot record 'v3'.


def build_sampling_params() -> SamplingParams:
    """One label token, constrained to LABELS.

    calibration: vLLM renamed constrained decoding — `guided_choice` on
    SamplingParams became `structured_outputs=StructuredOutputsParams(choice=)`
    around v0.11. Try the current spelling, fall back to the old one, so this
    runs on whichever version Kaggle's image ships.
    """
    try:
        from vllm.sampling_params import StructuredOutputsParams
    except ImportError:
        return SamplingParams(temperature=0, max_tokens=4, guided_choice=list(LABELS))
    return SamplingParams(
        temperature=0,
        max_tokens=4,
        structured_outputs=StructuredOutputsParams(choice=list(LABELS)),
    )


def load_template(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def read_rows(csv_path: str) -> list[dict]:
    with open(csv_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="labeling_batch_<date>.csv")
    ap.add_argument("--prompt", required=True, help="evals/prompts/jiwon_llama_v*.md")
    ap.add_argument("--output", required=True, help="labels_<date>.csv")
    ap.add_argument("--run-date", required=True, help="YYYY-MM-DD")
    args = ap.parse_args()

    template = load_template(args.prompt)
    # Fail before the GPU work, not after it: a missing marker discovered at
    # write time would throw away the whole run.
    prompt_version = parse_prompt_version(template)
    rows = read_rows(args.input)
    prompts = [render_prompt(text_for(r), template) for r in rows]

    llm = LLM(model=MODEL, quantization=QUANTIZATION, max_model_len=MAX_MODEL_LEN)
    params = build_sampling_params()
    # calibration: use .chat() so the Instruct chat template is applied.
    outputs = llm.chat([[{"role": "user", "content": p}] for p in prompts], params)

    meta = json.dumps(
        build_model_meta(MODEL, QUANTIZATION, prompt_version, args.run_date)
    )
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["raw_item_id", "label", "model_meta"])
        w.writeheader()
        for row, out in zip(rows, outputs, strict=True):
            label = validate_label(out.outputs[0].text)
            w.writerow(
                {"raw_item_id": row["raw_item_id"], "label": label, "model_meta": meta}
            )
    print(f"wrote {len(rows)} labels to {args.output}")


if __name__ == "__main__":
    main()
