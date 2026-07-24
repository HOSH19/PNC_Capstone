# Kaggle Llama 라벨링 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Kaggle 무료 GPU에서 Llama 3.1 8B(vLLM)로 export CSV의 기사를 리스크-방향 감성(`positive|negative|neutral`)으로 라벨링해 `labels_<date>.csv`를 만든다.

**Architecture:** 테스트 가능한 순수 로직(프롬프트 렌더링·라벨 검증·model_meta)은 repo 모듈 `scoring/labeling.py`에 두고 pytest로 검증한다. GPU 전용 글루(vLLM 로드·배치 추론)는 `labeling/kaggle_llama_labeling.py` thin 드라이버에 격리하며, 검증은 Kaggle 수동 스모크 테스트다. 텍스트 조립 규칙은 `pipeline/eligibility`에만 존재하고 라벨러가 재사용한다(DRY).

**Tech Stack:** repo 측 Python 3.13 + pytest(GPU 불필요). Kaggle 측 vLLM + Llama 3.1 8B Instruct(AWQ 4bit).

## Global Constraints

- **DRY**: 소스별 텍스트 조립 규칙은 `pipeline/eligibility`에만 둔다 — 라벨러는 `text_for(row)`를 import해 재사용, 절대 재정의 금지(train/serve 및 label/serve skew 방지와 같은 원칙).
- 라벨 값: `("positive", "negative", "neutral")` 소문자 3개 고정.
- 프롬프트 텍스트는 `evals/prompts/jiwon_llama.md`(git 버전 관리, 코드 아님), `{{ARTICLE}}` placeholder 포함.
- 라벨은 **기사 텍스트만으로**(결과 훔쳐보기 금지, leakage 방지) — 프롬프트에 반영.
- 코퍼스 영어 전용 → 프롬프트·few-shot 영어.
- **vLLM API는 버전 민감**(guided decoding 파라미터명, chat 템플릿 적용) — Kaggle에 설치된 vLLM 버전 기준으로 확인. Task 3의 게이트는 pytest가 아니라 **수동 스모크 테스트**.
- vLLM은 repo `requirements.txt`에 넣지 않는다(Kaggle 전용 환경).
- 테스트는 GPU·DB 불필요. 커밋 전 `pytest` 통과 + pre-commit(ruff). 커밋 메시지: 명령형 요약 + `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

---

### Task 1: eligibility에 text_for 노출 (DRY 준비)

라벨러가 텍스트 조립을 재정의하지 않도록, 현재 `check()` 내부에 있는 조립
로직을 공개 함수 `text_for(row)`로 추출하고 `check()`가 그것을 쓰게 리팩터.

**Files:**
- Modify: `pipeline/eligibility.py`
- Test: `tests/test_eligibility.py` (기존 파일에 케이스 추가)

**Interfaces:**
- Produces: `text_for(row: dict) -> str` — source 어댑터 규칙대로 텍스트 조립;
  미등록 source면 `UnknownSource`. 빈 결과면 `""` 반환.

- [ ] **Step 1: Write the failing test**

`tests/test_eligibility.py` 끝에 추가:

```python
def test_text_for_gdelt_returns_title():
    from pipeline.eligibility import text_for
    assert text_for(_gdelt(title="Deposit run at X")) == "Deposit run at X"


def test_text_for_edgar_joins_title_and_excerpt():
    from pipeline.eligibility import text_for
    assert text_for(_edgar()) == "Holding Co 8-K\nMaterial event disclosed."


def test_text_for_unknown_source_raises():
    from pipeline.eligibility import text_for, UnknownSource
    with pytest.raises(UnknownSource):
        text_for({"source": "foo", "title": "x", "meta": {}})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_eligibility.py -k text_for -v`
Expected: FAIL — `ImportError: cannot import name 'text_for'`.

- [ ] **Step 3: Refactor implementation**

`pipeline/eligibility.py`에서 `check()`를 아래처럼 바꾸고 `text_for`를 추가:

```python
def text_for(row: dict) -> str:
    """Assemble the scorable text for a row per its source adapter.

    The single definition of "what text represents this item" — imported by
    both check() and the labeling driver so they never drift.
    """
    try:
        adapter = ADAPTERS[row["source"]]
    except KeyError:
        raise UnknownSource(row["source"]) from None
    return "\n".join(str(row[f]) for f in adapter.text_fields if row.get(f)).strip()


def check(row: dict) -> Result:
    """Judge one raw_item row. Sole entry point shared by export and serving."""
    try:
        adapter = ADAPTERS[row["source"]]
    except KeyError:
        raise UnknownSource(row["source"]) from None

    if not adapter.is_english(row):
        return Result(False, reason="non_english")

    if is_syndication_noise(row):
        return Result(False, reason="syndication_noise")

    text = text_for(row)
    if not text:
        return Result(False, reason="empty_text")

    return Result(True, text=text)
```

- [ ] **Step 4: Run tests to verify all pass**

Run: `pytest tests/test_eligibility.py -v`
Expected: PASS — 9 passed (기존 6 + 신규 3).

- [ ] **Step 5: Commit**

```bash
git add pipeline/eligibility.py tests/test_eligibility.py
git commit -m "Extract text_for from eligibility.check for reuse

The labeling driver needs the same source-specific text assembly; expose it
as text_for so there is one definition instead of a copy that can drift.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: 프롬프트 파일 + labeling 순수 모듈

**Files:**
- Create: `evals/prompts/jiwon_llama.md`
- Create: `scoring/labeling.py`
- Test: `tests/test_labeling.py`

**Interfaces:**
- Consumes: `pipeline.eligibility.text_for` (Task 1).
- Produces:
  - `LABELS: tuple[str, ...]` = `("positive", "negative", "neutral")`
  - `ARTICLE_PLACEHOLDER: str` = `"{{ARTICLE}}"`
  - `render_prompt(article_text: str, template: str) -> str`
  - `validate_label(raw: str) -> str` — 정규화 후 `LABELS` 검증, 아니면 `ValueError`
  - `build_model_meta(model, quantization, prompt_version, run_date) -> dict`

- [ ] **Step 1: Create the prompt file**

`evals/prompts/jiwon_llama.md`:

```markdown
<!-- prompt_version: v1 -->
You label news articles about US banks by the RISK DIRECTION they imply for
the bank — not by the article's emotional tone.

- negative: implies the bank's risk is RISING / health worsening (losses,
  deposit outflows, enforcement actions, lawsuits, risk/finance executive
  exits, "exploring strategic alternatives", and other euphemistic distress
  signals).
- positive: implies risk FALLING / health improving (capital raises, consent
  orders lifted, earnings improvement, rating upgrades).
- neutral: no clear risk direction (routine announcements, product launches,
  branch openings, sponsorships, incidental mentions).

Examples:
Article: "Regional Bank reports third straight quarter of deposit outflows"
Label: negative
Article: "Community Bancorp says it is exploring strategic alternatives"
Label: negative
Article: "Pinnacle Bank's chief risk officer resigns after two years"
Label: negative
Article: "Federal Reserve lifts consent order against Midwest Bank"
Label: positive
Article: "Summit Bank raises $500M in capital, lifting its Tier 1 ratio"
Label: positive
Article: "Coastal Bank opens three new branches in the metro area"
Label: neutral

Now label this article. Answer with exactly one word: positive, negative,
or neutral.

Article: "{{ARTICLE}}"
Label:
```

- [ ] **Step 2: Write the failing test**

`tests/test_labeling.py`:

```python
import pytest

from scoring.labeling import (
    LABELS,
    build_model_meta,
    render_prompt,
    validate_label,
)


def test_render_prompt_inserts_article():
    out = render_prompt("Bank X fails", "before {{ARTICLE}} after")
    assert out == "before Bank X fails after"


def test_render_prompt_missing_placeholder_raises():
    with pytest.raises(ValueError):
        render_prompt("x", "no placeholder here")


def test_validate_label_normalizes():
    assert validate_label("Positive\n") == "positive"


def test_validate_label_rejects_unknown():
    with pytest.raises(ValueError):
        validate_label("maybe")


def test_labels_are_the_three_classes():
    assert LABELS == ("positive", "negative", "neutral")


def test_build_model_meta_shape():
    meta = build_model_meta("meta-llama/Llama-3.1-8B-Instruct", "awq-4bit",
                            "v1", "2026-07-21")
    assert meta == {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "quantization": "awq-4bit",
        "prompt_version": "v1",
        "run_date": "2026-07-21",
    }
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_labeling.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scoring.labeling'`.

- [ ] **Step 4: Write implementation**

`scoring/labeling.py`:

```python
"""Pure, testable core for the Kaggle Llama labeling notebook.

The Kaggle driver (labeling/kaggle_llama_labeling.py) is a thin GPU wrapper
around these functions; everything here runs and is tested without a GPU.
Prompt text lives in evals/prompts/*.md (git-versioned), not here.
"""

LABELS: tuple[str, ...] = ("positive", "negative", "neutral")
ARTICLE_PLACEHOLDER = "{{ARTICLE}}"


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


def build_model_meta(model: str, quantization: str, prompt_version: str,
                     run_date: str) -> dict:
    """Provenance recorded per label row (DESIGN Kaggle round-trip)."""
    return {
        "model": model,
        "quantization": quantization,
        "prompt_version": prompt_version,
        "run_date": run_date,
    }
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_labeling.py -v`
Expected: PASS — 6 passed.

- [ ] **Step 6: Commit**

```bash
git add evals/prompts/jiwon_llama.md scoring/labeling.py tests/test_labeling.py
git commit -m "Add labeling prompt and pure labeling helpers

Risk-direction 3-class prompt (git-versioned in evals/prompts) plus GPU-free
helpers: render_prompt, validate_label, build_model_meta. The Kaggle vLLM
driver wraps these.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Kaggle vLLM 드라이버 (GPU 전용, 수동 검증)

GPU가 필요해 repo에서 unit test 불가. 검증 = Kaggle 수동 스모크 테스트.

**Files:**
- Create: `labeling/kaggle_llama_labeling.py`
- Create: `labeling/README.md` (수동 실행 절차)

**Interfaces:**
- Consumes: `scoring.labeling`(Task 2), `pipeline.eligibility.text_for`(Task 1).

- [ ] **Step 1: Write the driver script**

`labeling/kaggle_llama_labeling.py`:

```python
"""Kaggle GPU driver: label the export CSV with Llama via vLLM.

Run inside a Kaggle notebook with GPU (T4 x2 or P100). This file is GPU-bound
glue only — pure logic is in scoring/labeling.py and pipeline/eligibility.py,
which must be importable (attach the repo, or upload pipeline/ + scoring/).

vLLM's guided-decoding and chat APIs are VERSION-SENSITIVE; if a call fails,
check the installed vLLM version and adjust the two marked lines.
"""

import argparse
import csv

from vllm import LLM, SamplingParams

from pipeline.eligibility import text_for
from scoring.labeling import build_model_meta, render_prompt, validate_label

MODEL = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"  # calibration: confirm on Kaggle
QUANTIZATION = "awq"
PROMPT_VERSION = "v1"


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
    params = SamplingParams(temperature=0, max_tokens=4,
                            guided_choice=list(("positive", "negative", "neutral")))
    # calibration: use .chat() so the Instruct chat template is applied.
    outputs = llm.chat([[{"role": "user", "content": p}] for p in prompts], params)

    meta = build_model_meta(MODEL, QUANTIZATION, PROMPT_VERSION, args.run_date)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["raw_item_id", "label", "model_meta"])
        w.writeheader()
        for row, out in zip(rows, outputs):
            label = validate_label(out.outputs[0].text)
            w.writerow({"raw_item_id": row["raw_item_id"], "label": label,
                        "model_meta": meta})
    print(f"wrote {len(rows)} labels to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the run procedure**

`labeling/README.md`:

```markdown
# Kaggle Llama 라벨링 — 실행 절차

1. Kaggle 노트북 생성 → Settings에서 **Accelerator = GPU (T4 x2 또는 P100)**.
2. export CSV(`labeling_batch_<date>.csv`)를 **비공개 Kaggle 데이터셋**으로
   업로드하고 노트북에 첨부.
3. 이 repo를 노트북에 첨부(또는 `pipeline/`, `scoring/`를 업로드)해
   import가 되게 한다.
4. 설치: `pip install vllm` (첫 실행은 가중치 다운로드로 수 분).
5. 실행:
   `python labeling/kaggle_llama_labeling.py --input labeling_batch_<date>.csv \
     --prompt evals/prompts/jiwon_llama.md --output labels_<date>.csv \
     --run-date <YYYY-MM-DD>`
6. `labels_<date>.csv` 다운로드 → repo import 스크립트로.

## 스모크 테스트 (첫 실행 필수)
- 20행짜리 mini CSV로 end-to-end 1회: 모든 행에 유효 라벨, 스키마
  (`raw_item_id, label, model_meta`) 일치, `model_meta` 채워짐.
- 같은 입력 재실행 시 라벨 동일(temperature=0 재현성).
- 클래스 분포가 상식적인지(전부 neutral 등 이상 없나).
```

- [ ] **Step 3: Lint the driver (GPU 불필요 부분만)**

Run: `ruff check labeling/kaggle_llama_labeling.py`
Expected: All checks passed (vLLM import는 미설치라도 ruff는 통과 — 실행 아님).

- [ ] **Step 4: Commit**

```bash
git add labeling/kaggle_llama_labeling.py labeling/README.md
git commit -m "Add Kaggle vLLM labeling driver and run procedure

GPU-bound glue: reads export CSV, batches Llama 3.1 8B (AWQ) via vLLM with
guided_choice over the three labels, writes labels_<date>.csv. Pure logic is
reused from scoring/labeling and eligibility.text_for. Verified by the manual
Kaggle smoke test in labeling/README.md (no repo-side GPU).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 5: Manual verification (Kaggle)**

`labeling/README.md`의 스모크 테스트를 Kaggle에서 수행. 실패 시 vLLM 버전에
맞춰 표시된 두 줄(guided decoding / chat)만 조정.

---

## Self-Review

**1. Spec coverage** — 스펙 요구 매핑:
- 런타임 vLLM + guided decoding → Task 3 드라이버 ✅
- 모델/양자화(Llama 3.1 8B AWQ) → Task 3 `MODEL`/`QUANTIZATION` ✅
- 리스크-방향 3-클래스 정의 + few-shot → Task 2 프롬프트 파일 ✅
- 출력 유효성 보장(guided_choice + validate_label) → Task 2/3 ✅
- 텍스트 조립 = eligibility 재사용(DRY) → Task 1 `text_for` ✅
- model_meta 스키마 → Task 2 `build_model_meta` ✅
- 노트북 흐름·수동 단계·재현성(temperature=0) → Task 3 드라이버 + README ✅
- 프롬프트를 `evals/prompts/`에 git 버전 관리 → Task 2 ✅
- 비목표(export/import·Gemini·파인튜닝) → 계획에서 제외 ✅

**2. Placeholder scan** — 완성 코드/절차 포함. `MODEL`과 vLLM 두 줄의
"calibration" 주석은 스펙이 명시한 **환경 의존 검증 노브**(placeholder 아님) —
수동 스모크 테스트가 게이트. ✅

**3. Type consistency** — `text_for`, `render_prompt`, `validate_label`,
`build_model_meta`, `LABELS`가 정의(Task 1·2)와 사용(Task 3)에서 일치.
라벨 3-클래스 문자열 동일. ✅

이슈 없음.
