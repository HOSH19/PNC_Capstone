# 라벨링 가이드 — 사람 검증용 (팀원용)

> 이 문서 하나만 보고 라벨링할 수 있게 썼습니다. 코드 필요 없음.

## 뭐 하는 건가요? (30초)

은행 뉴스/공시 기사 수십 개를 읽고, 각각 **positive / negative / neutral**
하나를 고릅니다. 이건 AI(Llama)가 매긴 라벨이 믿을 만한지 확인하는
**사람 기준 정답지**가 됩니다. 엑셀/구글시트에서 `label` 칸만 채우면 끝.

대상 기사는 **뉴스(gdelt)와 기업 공시(edgar)** 두 종류입니다.

## 절대 규칙 3개

1. **"말투"가 아니라 "은행 리스크 방향"으로 판단.** (아래 정의)
2. **기사 텍스트만 보고 판단.** 이 은행이 나중에 어떻게 됐는지 검색·추측 ❌.
3. **정말 헷갈리면 → `neutral`.** 찍지 말 것. (`comment` 칸에 왜 헷갈렸는지
   한 줄 남겨주면 최고)

## 3-클래스 정의 (리스크 방향)

- **negative** — 은행 위험이 **커진다** 싶으면.
  손실, 예금 빠짐, 규제 제재·벌금, 소송, 리스크/재무 임원 사임,
  "전략적 대안 모색" 같은 **완곡한 위기 신호**까지 포함.
- **positive** — 은행이 **더 튼튼해진다** 싶으면.
  자본 확충, 규제 제재 **해제**, 실적 개선, 신용등급 상향.
- **neutral** — 위험 방향이 뚜렷하지 않음.
  일상 공지, 상품 출시, 지점 개설, 스폰서십, 은행이 곁다리로만 언급된 기사.

## 헷갈리기 쉬운 함정 (여기서 실력 남 — 꼭 읽기)

말투와 리스크가 **어긋나는** 경우들:

| 기사 | 정답 | 왜 |
|---|---|---|
| "전략적 대안 모색 중" | **negative** | 말투는 담담해도 매각/부실 전조 |
| "규제당국, 동의명령 해제" | **positive** | '규제' 단어 있어도 위험은 **내려감** |
| "배당 동결(그대로 유지)" | **neutral** | 나쁜 소식 아님. *삭감*이면 negative |
| "은행, 마라톤 후원" | **neutral** | 리스크와 무관 |
| "OO캐피탈, 이 은행 주식 1,200주 매수" | **neutral** | 남이 우리 은행 주식 산 것 = 리스크 신호 아님 |

## 어떻게 채우나

1. 받은 CSV(또는 시트)를 엽니다. 컬럼:
   `id, source, title, text_excerpt, label, comment`.
2. 각 행의 `title`을 읽습니다. `source`가 `edgar`면 `text_excerpt`(공시 발췌)도 같이.
3. `label` 칸에 **`positive` / `negative` / `neutral`** 중 하나를 **소문자**로 적습니다.
4. 헷갈렸으면 `comment`에 한 줄 (선택).
5. **모든 행을 채웁니다.** 빈 칸 ❌.
6. 저장해서 Jiwon에게 돌려줍니다.

## 채운 예시 (이렇게 하면 됨)

| id | source | title | label | comment |
|---|---|---|---|---|
| 1001 | gdelt | Regional Bank reports third straight quarter of deposit outflows | negative | 예금 이탈 |
| 1005 | gdelt | Federal Reserve lifts consent order against Midwest Bank | positive | 제재 해제라 호재 |
| 1017 | edgar | ...dividend of $0.20, unchanged from prior quarter | neutral | 동결이라 방향 없음 |

## 하지 말 것

- 코드 작성 ❌ / label에 pos·neg·neu 외 다른 말 ❌
- 인터넷에서 "이 은행 이후 결과" 찾아보기 ❌ (규칙 2)
- 헷갈린다고 빈칸 두기 ❌ → `neutral` + comment

막히면 Jiwon에게 물어보세요.
