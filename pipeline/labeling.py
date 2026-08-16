"""Pure, testable core for the Kaggle Llama labeling notebook.

The Kaggle driver (labeling/kaggle_llama_labeling.py) is a thin GPU wrapper
around these functions; everything here runs and is tested without a GPU.
Prompt text lives in evals/prompts/*.md (git-versioned), not here — and so
does its version marker, which parse_prompt_version reads.
"""

import re

LABELS: tuple[str, ...] = ("positive", "negative", "neutral")
ARTICLE_PLACEHOLDER = "{{ARTICLE}}"
VERSION_MARKER = re.compile(r"<!--\s*prompt_version:\s*(\S+?)\s*-->")


def parse_prompt_version(template: str) -> str:
    """Read the version marker the prompt file declares about itself.

    The prompt text and its version are one artifact: a driver that carries
    its own PROMPT_VERSION constant will keep writing 'v2' into model_meta
    after someone points it at a v3 file, and the provenance silently lies.
    """
    m = VERSION_MARKER.search(template)
    if not m:
        raise ValueError("prompt file has no <!-- prompt_version: ... --> marker")
    return m.group(1)


def render_prompt(article_text: str, template: str) -> str:
    """Insert one article into the prompt template."""
    if ARTICLE_PLACEHOLDER not in template:
        raise ValueError(f"prompt template missing {ARTICLE_PLACEHOLDER}")
    return template.replace(ARTICLE_PLACEHOLDER, article_text)


def validate_label(raw: str) -> str:
    """Normalize a model output to one of LABELS, or raise."""
    label = raw.strip().lower()
    if label not in LABELS:
        raise ValueError(f"invalid label: {raw!r}")
    return label


def build_model_meta(
    model: str, quantization: str, prompt_version: str, run_date: str
) -> dict:
    """Provenance recorded per label row (DESIGN Kaggle round-trip)."""
    return {
        "model": model,
        "quantization": quantization,
        "prompt_version": prompt_version,
        "run_date": run_date,
    }
