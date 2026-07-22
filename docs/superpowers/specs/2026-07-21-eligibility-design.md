# eligibility 모듈 설계 스펙

- **날짜**: 2026-07-21
- **상태**: 승인됨 (구현 대기)
- **관련 문서**: [scoring/DESIGN.md](../../../scoring/DESIGN.md), [scoring/INTEGRATION.md](../../../scoring/INTEGRATION.md), `db/migrations/011_scoring_tables.sql`

## 배경 / 문제

스코어링 파이프라인은 두 지점에서 "어떤 `raw_item`을 채점 대상으로 볼지"를
판단한다:

1. **라벨링 export** (Stage 1) — 학습 코퍼스로 뽑을 행 선정.
2. **일 배치 서빙** (Stage 3) — 매일 `finbert_status='pending'` 행 채점.

이 판단(언어·노이즈 필터)을 두 곳이 각자 구현하면 "학습 때 본 분포"와
"서빙에서 채점하는 분포"가 조용히 어긋난다(**train/serve skew**). 따라서
**단일 eligibility 모듈**을 양쪽이 호출하는 single source of truth로 둔다.

DESIGN 계약: "eligibility = 언어 + 노이즈 필터, 그 외 없음." Risk lexicon은
선정 기준이 아니다(중요 리스크 기사에 명시적 부정어가 없는 경우가 많음).

## 목표 (v1 범위)

- 소스 무관하게 호출하는 단일 함수 `check(row)` 제공.
- **언어 필터** 실제 구현 (이미 저장된 값 사용, 새 의존성 없음).
- **노이즈 필터**는 호출 지점(훅)만 두고 predicate는 지연 — export dry-run으로
  실제 분포를 본 뒤 확정 (13F/holdings 스팸, EDA 기준 GDELT의 ~14%).
- 소스별 텍스트 필드 추출을 어댑터로 캡슐화.

## 비목표 (v1에서 제외)

- 실제 신디케이션 노이즈 predicate (데이터 기반 튜닝으로 지연).
- 소스 간 dedup — per-row 판단이 아니라 export 배치의 집합 연산이므로 이 모듈
  밖(export 스크립트).
- 비영어 처리(번역 등) — Phase 3.

## 인터페이스 계약

```python
# pipeline/eligibility.py

class UnknownSource(Exception):
    """raw_item.source에 대응하는 어댑터가 없음 → 등록 필요."""

@dataclass(frozen=True)
class Result:
    eligible: bool
    reason: str | None   # eligible=False일 때만; {"non_english",
                         #   "syndication_noise", "empty_text"}
    text: str | None     # eligible=True일 때만; FinBERT에 넣을 문자열

def check(row: dict) -> Result:
    """raw_item dict 하나를 판정. export와 서빙이 공유하는 유일한 진입점."""
```

- `row`는 `db.py`의 dict_row가 준 `raw_item` 한 행 (키: `source`, `title`,
  `text_excerpt`, `meta`, ...).
- 미등록 소스는 `UnknownSource`를 던진다 — "등록 안 하면 pending에 방치"
  계약을 조용한 skip이 아니라 **시끄러운 신호**로 만든다. 호출자는 경고
  로깅 후 그 행을 `pending`으로 남긴다(서빙) / 건너뛴다(export).

## 소스 어댑터 레지스트리

팀원이 새 텍스트 소스를 채점 대상으로 만드는 지점(INTEGRATION Pattern B).

```python
@dataclass(frozen=True)
class Adapter:
    text_fields: tuple[str, ...]        # 이어붙일 텍스트 컬럼
    is_english: Callable[[dict], bool]  # 소스별 언어 판정

ADAPTERS: dict[str, Adapter] = {
    "gdelt": Adapter(
        text_fields=("title",),                    # 본문 미수집, 제목만
        is_english=lambda r: (r["meta"] or {}).get("language") == "English",
    ),
    "edgar": Adapter(
        text_fields=("title", "text_excerpt"),     # 8-K 제목 + 발췌
        is_english=lambda r: True,                 # SEC 공시는 항상 영어
    ),
}
```

**보정 노브(calibration)**: GDELT `meta.language`의 정확한 문자열 값
(`"English"`)은 실제 데이터(2026-07-16 EDA)에서 확인된 값과 일치해야 한다.
값 표기가 다르면 이 람다 한 줄만 조정.

## check() 로직 (순서)

1. `ADAPTERS[row["source"]]` 조회 → 없으면 `UnknownSource(row["source"])`.
2. **언어**: `adapter.is_english(row)`가 False → `Result(False, "non_english", None)`.
3. **노이즈(v1 = 훅)**: `is_syndication_noise(row)` 호출.
   - v1 구현은 `return False` (결정 A: predicate 지연).
   - `ponytail:` 주석으로 지연을 명시하고, export dry-run 후 predicate 채울
     자리임을 표시.
   - True면 `Result(False, "syndication_noise", None)`.
4. **텍스트 조립**: `text_fields`의 non-empty 값을 `"\n"`으로 이어붙임.
   빈 문자열이면 `Result(False, "empty_text", None)`.
5. 통과 → `Result(True, None, text)`.

## 소비자 사용법

- **Export (Stage 1)**: 후보 행마다 `check(r)`; `eligible`인 행만 CSV로,
  `r.text`를 텍스트 컬럼에. dry-run은 소스별 eligible/skip 카운트를 6단계
  퍼널로 보고.
- **서빙 (Stage 3)**: `pending` 행마다 `r = check(row)`.
  - 부적격 → `finbert_status='skipped'`, `last_error=r.reason`.
  - 적격 → `r.text`를 FinBERT에, 결과를 `item_score`에 upsert 후
    `finbert_status='done'`.
  - `UnknownSource` → 경고 로깅, 행은 `pending` 유지 (어댑터 등록 유도).

## 에러 처리

| 상황 | 동작 |
|---|---|
| 미등록 소스 | `UnknownSource` 예외; 호출자가 pending 유지 + 경고 |
| 비영어 | `reason="non_english"`, 서빙은 `skipped` |
| 빈 텍스트 | `reason="empty_text"`, 서빙은 `skipped` |
| 노이즈(v1) | 항상 통과(훅만); predicate 채워지면 `skipped` |

## 테스트 계획

`tests/test_eligibility.py` — pytest, DB 불필요(dict만), 표 기반 케이스:

1. gdelt + `meta.language="English"` → eligible, `text == title`.
2. gdelt + `meta.language="Spanish"` → `non_english`.
3. edgar → eligible, `text == title + "\n" + text_excerpt`.
4. gdelt + `title=None` → `empty_text`.
5. 미등록 소스(`"foo"`) → `UnknownSource` raise.
6. 노이즈 훅이 존재하고 v1에서 항상 False임을 확인(지연 문서화).

## 지연/후속 (Deferred)

- 신디케이션 노이즈 predicate: export dry-run으로 domain·title 패턴·
  `n_duplicates` 분포 확인 후 `is_syndication_noise`에 구현.
- 소스 간 dedup: export 스크립트(별도 작업).
- 비영어 처리: Phase 3.
