"""Kaggle GPU driver: label the export CSV with Llama via vLLM.

This file is GPU-bound glue only — pure logic is in pipeline/labeling.py and
pipeline/eligibility.py, which it reuses. It is NOT part of the ingestion
pipeline and is never imported by it; it runs by hand in a Kaggle notebook.

Run procedure (Kaggle):
  1. New Kaggle notebook → Settings → Accelerator = GPU (T4 x2 or P100).
  2. Upload the export CSV (labeling_batch_<date>.csv) as a PRIVATE Kaggle
     dataset and attach it.
  3. Attach this repo (or upload pipeline/) so the imports resolve.
  4. pip install vllm  (first run downloads weights, a few minutes).
  5. python pipeline/kaggle_llama_labeling.py \
       --input labeling_batch_<date>.csv \
       --prompt evals/prompts/jiwon_llama.md \
       --output labels_<date>.csv --run-date <YYYY-MM-DD>
  6. Download labels_<date>.csv → repo import script.

Smoke test (required on first run):
  - 20-row mini CSV end-to-end: every row gets a valid label, schema is
    (raw_item_id, label, model_meta), model_meta is JSON.
  - Re-run with same input gives identical labels (temperature=0).
  - Class distribution is sane (not all neutral).

vLLM's guided-decoding and chat APIs are VERSION-SENSITIVE. If a call fails,
check the installed vLLM version and adjust the two lines marked `calibration`
(newer vLLM uses guided_decoding=GuidedDecodingParams(choice=...); .chat()
may need a manual chat-template apply on .generate()). Confirm the AWQ
checkpoint path is reachable from Kaggle too.
"""

import argparse
import csv
import json

from vllm import LLM, SamplingParams

from pipeline.eligibility import text_for
from pipeline.labeling import LABELS, build_model_meta, render_prompt, validate_label

# calibration: confirm this AWQ checkpoint is reachable on Kaggle.
MODEL = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
QUANTIZATION = "awq"
PROMPT_VERSION = "v2"


def load_template(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def read_rows(csv_path: str) -> list[dict]:
    with open(csv_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="labeling_batch_<date>.csv")
    ap.add_argument("--prompt", required=True, help="evals/prompts/jiwon_llama.md")
    ap.add_argument("--output", required=True, help="labels_<date>.csv")
    ap.add_argument("--run-date", required=True, help="YYYY-MM-DD")
    args = ap.parse_args()

    template = load_template(args.prompt)
    rows = read_rows(args.input)
    prompts = [render_prompt(text_for(r), template) for r in rows]

    llm = LLM(model=MODEL, quantization=QUANTIZATION)
    # calibration: guided-decoding param name varies by vLLM version.
    params = SamplingParams(temperature=0, max_tokens=4, guided_choice=list(LABELS))
    # calibration: use .chat() so the Instruct chat template is applied.
    outputs = llm.chat([[{"role": "user", "content": p}] for p in prompts], params)

    meta = json.dumps(
        build_model_meta(MODEL, QUANTIZATION, PROMPT_VERSION, args.run_date)
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
