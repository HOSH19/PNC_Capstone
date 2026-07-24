# eligibility 모듈 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 라벨링 export와 일 배치 서빙이 공유하는 단일 eligibility 필터(`check(row)`)를 만든다 — 무엇을 채점 대상으로 볼지 한 곳에서 판정해 train/serve skew를 차단한다.

**Architecture:** `pipeline/eligibility.py`에 순수 함수 하나(`check`)와 소스별 어댑터 레지스트리(`ADAPTERS`)를 둔다. DB·I/O 없음 — `raw_item` dict를 받아 `Result(eligible, reason, text)`를 반환. v1은 언어 필터만 실제 구현하고, 신디케이션 노이즈는 훅(`is_syndication_noise`, 현재 `return False`)만 둔다.

**Tech Stack:** Python 3.13, stdlib(`dataclasses`, `collections.abc`), pytest. 새 의존성 없음.

## Global Constraints

- Python 3.13, **stdlib만** — 새 의존성 추가 금지.
- DB·네트워크 없음 — `check`는 dict in / `Result` out인 순수 함수.
- 테스트는 `tests/test_eligibility.py`, pytest, DB 불필요(dict 픽스처만). 기존 관례(`tests/test_<module>.py`).
- `pipeline/` 평평한 레이아웃 유지 (db.py·http.py와 형제).
- 단일 진입점 `check(row)` — export와 서빙이 **같이** 호출 (single source of truth).
- 노이즈 predicate는 **지연**(결정 A, 2026-07-21): 훅은 `return False` + `ponytail:` 주석. export dry-run 후 채운다.
- 커밋 전 `pytest tests/test_eligibility.py` 통과 확인. pre-commit이 ruff format/check 실행.
- 커밋 메시지: 저장소 관례(명령형 대문자 요약) + `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

---

### Task 1: eligibility 모듈 + 테스트

**Files:**
- Create: `pipeline/eligibility.py`
- Test: `tests/test_eligibility.py`

**Interfaces:**
- Consumes: `raw_item` dict (키: `source`, `title`, `text_excerpt`, `meta`) — `db.py`의 dict_row가 주는 행 형태.
- Produces (후속 export/서빙 작업이 의존):
  - `check(row: dict) -> Result`
  - `Result(eligible: bool, reason: str | None, text: str | None)` — frozen dataclass
  - `UnknownSource(Exception)` — 미등록 소스일 때 raise
  - `is_syndication_noise(row: dict) -> bool` — v1은 항상 False
  - `ADAPTERS: dict[str, Adapter]` — 소스 등록 지점(Pattern B)

- [ ] **Step 1: Write the failing test**

`tests/test_eligibility.py`:

```python
import pytest

from pipeline.eligibility import Result, UnknownSource, check, is_syndication_noise


def _gdelt(title="Bank X under investigation", language="English", **kw):
    row = {"source": "gdelt", "title": title, "text_excerpt": None,
           "meta": {"language": language}}
    row.update(kw)
    return row


def _edgar(title="Holding Co 8-K", excerpt="Material event disclosed."):
    return {"source": "edgar", "title": title, "text_excerpt": excerpt, "meta": {}}


def test_gdelt_english_eligible():
    assert check(_gdelt()) == Result(True, text="Bank X under investigation")


def test_gdelt_non_english_skipped():
    r = check(_gdelt(language="Spanish"))
    assert r.eligible is False
    assert r.reason == "non_english"


def test_edgar_eligible_joins_title_and_excerpt():
    r = check(_edgar())
    assert r.eligible is True
    assert r.text == "Holding Co 8-K\nMaterial event disclosed."


def test_gdelt_empty_title_skipped():
    r = check(_gdelt(title=None))
    assert r.eligible is False
    assert r.reason == "empty_text"


def test_unknown_source_raises():
    with pytest.raises(UnknownSource):
        check({"source": "foo", "title": "x", "meta": {}})


def test_noise_hook_present_but_deferred():
    # v1: 훅은 존재하되 항상 False (predicate는 export dry-run 후 확정)
    assert is_syndication_noise(_gdelt()) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_eligibility.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pipeline.eligibility'` (또는 import 에러).

- [ ] **Step 3: Write minimal implementation**

`pipeline/eligibility.py`:

```python
"""Shared eligibility filter: which raw_item rows get scored.

Called by BOTH labeling export (Stage 1) and the daily serving batch
(Stage 3), so the two see the same distribution — one source of truth
prevents train/serve skew. See scoring/DESIGN.md.

v1 implements the language filter. The syndication-noise predicate is a
hook only (returns False) until the export dry-run reveals the real
distribution.
"""

from collections.abc import Callable
from dataclasses import dataclass


class UnknownSource(Exception):
    """No adapter for a raw_item.source — register one (INTEGRATION Pattern B)."""


@dataclass(frozen=True)
class Result:
    eligible: bool
    reason: str | None = None   # {"non_english", "syndication_noise", "empty_text"}
    text: str | None = None     # text to score, set when eligible


@dataclass(frozen=True)
class Adapter:
    text_fields: tuple[str, ...]
    is_english: Callable[[dict], bool]


ADAPTERS: dict[str, Adapter] = {
    "gdelt": Adapter(
        text_fields=("title",),                    # body not collected, title only
        is_english=lambda r: (r.get("meta") or {}).get("language") == "English",
    ),
    "edgar": Adapter(
        text_fields=("title", "text_excerpt"),     # 8-K title + excerpt
        is_english=lambda r: True,                 # SEC filings are always English
    ),
}


def is_syndication_noise(row: dict) -> bool:
    # ponytail: v1 hook only — the 13F/holdings spam predicate (EDA: ~14% of
    # GDELT) is deferred until the export dry-run shows the real domain/title/
    # n_duplicates distribution. Fill it in here then; export and serving both
    # pick it up for free. See scoring/DESIGN.md decision 2026-07-21.
    return False


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

    text = "\n".join(str(row[f]) for f in adapter.text_fields if row.get(f)).strip()
    if not text:
        return Result(False, reason="empty_text")

    return Result(True, text=text)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_eligibility.py -v`
Expected: PASS — 6 passed.

- [ ] **Step 5: Commit**

```bash
git add pipeline/eligibility.py tests/test_eligibility.py
git commit -m "Add eligibility filter module for scoring

Single check(row) shared by labeling export and serving batch to avoid
train/serve skew. v1: language filter (reads stored meta.language); the
syndication-noise predicate is a deferred hook (returns False) pending the
export dry-run. Sources register via the ADAPTERS registry.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**1. Spec coverage** — 스펙 각 요구와 태스크 매핑:
- 단일 `check(row)` 진입점 → Step 3 ✅
- 언어 필터 실제 구현(저장된 `meta.language`) → `ADAPTERS`/`is_english` ✅
- 노이즈 훅만(결정 A, `return False` + ponytail 주석) → `is_syndication_noise` ✅
- 소스 어댑터 레지스트리(Pattern B) → `ADAPTERS`/`Adapter` ✅
- 미등록 소스 → `UnknownSource` → Step 3 + test ✅
- 텍스트 조립(non-empty join) / `empty_text` → Step 3 + test ✅
- `Result(eligible, reason, text)` 계약 → dataclass + tests ✅
- 테스트 6케이스(스펙 "테스트 계획") → Step 1 ✅
- 비목표(실제 노이즈 predicate, 소스 간 dedup, 비영어 처리) → 계획에서 제외 ✅ (의도적)

**2. Placeholder scan** — "TBD/TODO/appropriate error handling" 등 없음. `is_syndication_noise`의 `return False`는 스펙이 명시한 의도된 v1 동작(placeholder 아님). 모든 코드 스텝에 완성 코드 포함. ✅

**3. Type consistency** — `check`/`Result`/`UnknownSource`/`is_syndication_noise`/`ADAPTERS`/`Adapter` 이름·시그니처가 Interfaces 블록, 구현(Step 3), 테스트(Step 1)에서 전부 일치. `reason` 값 집합 {non_english, syndication_noise, empty_text} 동일. ✅

이슈 없음.
