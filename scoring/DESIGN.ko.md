# scoring — 설계 (Phase 2)

> 영어 원본: [DESIGN.md](DESIGN.md) — 두 문서가 어긋나면 영어본이 기준.

[README.md](README.md)의 자매 문서. 2026-07-12 멘토와 합의한 방법론:
**LLM 보조 라벨링 → BERT 계열 모델 파인튜닝 → 일 단위 배치 예측**, 키워드
수준 explainability 포함. 이 문서는 구체적인 계약(contract)을 고정한다;
구현은 팀 리뷰 이후에 시작한다.

## 라벨을 매기는 작업은 하나가 아니라 셋이다

셋 다 결과물이 같은 3클래스(`positive | negative | neutral`)라서 헷갈리기
쉽다. 누가 만드는지, 무엇을 위한 것인지가 다르다:

| | 주체 | 입력 | 목적 | 상태 |
|---|---|---|---|---|
| ① **라벨링** | Llama (Kaggle GPU) | 수집된 8,360행 | FinBERT 학습용 정답지 | 완료 — 2026-07-22 배치 |
| ② **검증** | 팀원 5명 × 50행 | 250행 gold slice | ①이 믿을 만한가? (품질 게이트) | 진행 중 |
| ③ **스코어링** | 파인튜닝된 FinBERT (서빙) | 매일 새로 들어오는 행 | 대시보드에 나갈 리스크 점수 | 미구현 |

①과 ②는 하나의 질문만 던진다 — *이 텍스트가 함의하는 방향은?* — 이는 은행과
무관하며 그래서 맞다. ③에는 단계가 하나 더 있다: 결과가 **은행별로 합산**되고,
그 합산 지점이 은행 귀속이 문제가 되는 유일한 곳이다.

    ① ②   텍스트 → 방향                    은행 무관; 귀속 문제 없음
    ③     텍스트 → 방향 → 은행별 합산       여기서 귀속이 필요

이 문서 전반의 용어: **라벨링** = ①, **검증** = ②, **스코어링** = ③.
"은행 귀속" 섹션은 ③에 대한 것이며, 라벨링 배치나 진행 중인 검증 작업에는
아무 영향을 주지 않는다.

## 결정 로그

| 날짜 | 결정 | 근거 |
|---|---|---|
| 2026-07-15 | 라벨러: **Kaggle 무료 GPU의 Llama, 처음엔 단독**. Gemini API는 비상용 challenger로, 초기에는 셋업하지 않음. | Kaggle 환경은 FinBERT 파인튜닝(stage 2)에 재사용됨; API 비용/ToS/rate-limit 의존성 없음. Gemini 투입 조건은 "품질 게이트" 참조. |
| 2026-07-15 | 저장소: **새 테이블 2개** (`item_label`, `item_score`), `raw_item`은 무변경. | 라벨링 출력은 (아이템, 라벨러)별 이력(champion-challenger + 휴먼 행이 공존해야 함); 서빙 출력은 아이템당 최신 점수 하나. 수명주기가 다름 → 테이블 분리. 마이그레이션: `db/migrations/011_scoring_tables.sql`. |
| 2026-07-15 | 이 문서의 범위: 설계 + 스키마 계약만, 실행 코드는 아직 없음. | 팀원들의 데이터 소스 연동과 scoring 설계가 병렬로 진행될 수 있게 함. |
| 2026-07-17 | Risk lexicon은 **서술용 지표일 뿐 — 라벨링 eligibility 필터로 절대 사용하지 않음**. eligibility = 언어 + 노이즈 필터, 그 외 없음. | 멘토 피드백: 중요한 은행 리스크 기사에는 명백한 부정 단어가 없는 경우가 많음 ("bank explores strategic alternatives", "deposits fall for third consecutive quarter", "chief risk officer departs"); lexicon 필터는 바로 그런 기사들을 걸러내 버림. |
| 2026-07-17 | 코퍼스 집계는 **6단계 퍼널**을 사용; export dry-run이 6개 카운트를 모두 보고해야 함. | 멘토 피드백: 총량만 보면 소스 간 중복이 가려짐 (같은 기사가 GDELT / Alpha Vantage / NewsAPI / RSS / 신디케이션 도메인으로 유입); 퍼널이 실제 작업셋 크기를 드러냄. 단계는 아래 "대상 선정" 참조. |
| 2026-07-21 | 데이터 소스를 **3개 질문(재현성·귀속·지속성)**으로 티어링; NewsAPI / Alpha Vantage / Benzinga / Reuters(UCLA) / Company IR / yfinance는 **제외**. | 아래 "데이터 소스 선정 기준" 참조. 뉴스API 제외의 결정적 근거는 쿼터가 아니라 **과거 복원 불가**(NewsAPI 무료 lookback ~1개월 → 백테스트 학습 불가). |
| 2026-07-25 | **은행 귀속(attribution)은 `eligibility`가 아니라 집계 레이어에서 거른다.** eligible한 아이템은 전부 라벨링·스코어링하되, 그 점수를 원래 붙어있던 은행에 반영할지는 별도 체크가 정한다. | 2026-07-22 배치 실측: GDELT 제목이 자기 `bank_id` 은행명을 담고 있는 경우는 **7,756건 중 726건(9.4%)**뿐이고, gold slice의 방향성 행 **30건 중 21건(70%)**이 엉뚱한 은행에 귀속돼 있다. 이걸 `eligibility`에서 거르면 가짜 신호는 잡히지만 **학습 코퍼스의 ~90%가 날아간다** — 게다가 그 행들은 *유효한* 텍스트→방향 학습 예시다("Bandhan Bank Q1 폭락 → negative"는 은행 악재 헤드라인이 어떻게 생겼는지 가르쳐준다; 다만 JPMorgan의 악재가 아닐 뿐). 두 축을 분리하면 코퍼스를 지키면서 가짜 신호만 제거된다. 단 **처리 방식에 따른 분리**에 유의: 분석가/보유자 와이어 템플릿은 *다른* 문제다 — 방향성 라벨이 구조적으로 다른 회사의 것이므로 귀속만 막는 게 아니라 Stage 1(대상 선정 3번)에서 **폐기**한다. 귀속 단계로 넘기는 것은 "제목에 자기 은행명이 없는" 부류뿐이다. 아래 "은행 귀속" 참조. |
| 2026-07-24 | **`bank_id`는 provenance(출처)지 모델 feature가 아님.** FinBERT는 `text`(title, EDGAR는 title+excerpt) → 방향만 스코어링; 은행 귀속은 `raw_item.bank_id`(그 행을 fetch한 쿼리)로 **집계 레이어**에서 붙이며 모델 입력이 아니다. text-only 학습/서빙은 의도된 설계이지 누락이 아니다. | 리스크 방향은 대체로 은행 무관 — "deposit outflows"는 어느 은행이든 negative — 이라 `bank_id`를 입력에 넣어도 분류가 개선되지 않는다. 진짜 실패모드는 *귀속(attribution)*이다: 은행 X 쿼리로 fetch됐지만 제목은 엔티티 Y 얘기인 경우(은행이 분석가/보유자/CEO인 케이스 — "TD가 Louisiana-Pacific 목표가 제시", "HSBC가 MSC에 투자", "Dimon이 영국에 경고"). 이는 upstream relevance 문제(은행이 주어인가?)이지 작은 encoder가 노이즈 라벨로 안정적으로 배우는 게 아니다; 사람 라벨이 이미 이들을 `neutral`로 찍어둔다. relevance 게이트로 못 막는 귀속 오류가 층화 eval에서 확인될 때만 재검토. |

## 데이터 소스 선정 기준

우리 모델은 결국 이런 문장을 학습한다:

> "2019년의 A은행이 이런 상태였는데 → 이후 부실해졌다."

그래서 데이터 소스를 고를 때 "좋아 보이는가"가 아니라 **"이 문장을 만들
수 있는 데이터인가"**를 따진다. 이 판단을 3개 질문으로 나눴다.

**질문 1. 과거를 그때 모습 그대로 되살릴 수 있나? (재현성)**
모델은 "과거 시점의 상태 → 그 이후 결과"를 보고 배운다. 그러려면 2019년
데이터를 2019년 모습 그대로 꺼낼 수 있어야 한다.
- 됨: Call Report·규제 조치·FRED는 발표된 값이 영구 보관된다 → 언제든 그때로 되돌아간다.
- 안 됨: NewsAPI 무료 티어는 최근 ~1개월 기사만 준다. 2019년 기사를 못
  꺼내니 과거로 학습하는 것 자체가 불가능하다. (뉴스API를 뺀 **진짜 이유** —
  쿼터가 아니라 이것)

**질문 2. 이게 어느 은행 데이터인지 확실한가? (귀속)**
모든 데이터는 결국 은행 하나에 정확히 붙어야 한다.
- 확실: 규제 데이터는 은행 고유번호(RSSD/CERT)를 달고 발행된다 → 100% 이 은행 것.
- 흐릿: 뉴스 기사에 "PNC"가 나와도 그게 리스크 얘긴지 스포츠 스폰서 얘긴지
  기계가 헷갈린다.

**질문 3. 공짜로, 안 끊기고, 합법으로 계속 받을 수 있나? (지속성)**
파이프라인은 매일 자동으로 돈다. 소스가 중간에 막히면 안 된다.
- 안정: 정부 API(FDIC/Fed/OCC/FRED/SEC)는 무료 + 공식 + 재배포 자유.
- 불안: Benzinga(유료) · yfinance(비공식 스크래핑, 자주 깨짐) ·
  Reuters(도서관 "되면")는 언제 끊길지 모른다.

세 질문을 다 통과할수록 모델의 뼈대에 가깝고, 하나라도 걸리면 후순위거나
제외다. FDIC BankFind을 특별히 뼈대로 두는 건 질문 2 때문 — 은행 고유번호의
기준표라, 나머지 데이터를 은행별로 정확히 이어 붙여 주는 역할을 한다.

### 티어링

**Tier 1 — 채택** (정부가 은행별로 발행·영구 보관, 3개 질문 전부 통과)

| 소스 | 역할 |
|---|---|
| FFIEC Call Reports | 재무 피처 근간 (이미 fundamentals로 사용 중) |
| FDIC BankFind API | 엔티티 레지스트리 / 조인키 |
| FDIC Failed Bank List | 정답 라벨 (부실 ground truth) |
| FDIC Enforcement Orders | 규제 리스크 |
| OCC Enforcement Actions | 규제 리스크 |
| Federal Reserve Enforcement | 규제 리스크 (이미 인제스트) |

**Tier 2 — 보조** (3개 질문은 통과하나 커버리지가 부분적 → 후순위 추가)

| 소스 | 비고 |
|---|---|
| FRED API | 매크로 컨트롤(금리·실업률·수익률곡선). 은행별이 아닌 경제 배경 |
| CFPB Complaints | 은행별 소비자 불만 — 조기경보 신호 |
| SEC EDGAR | 8-K/10-Q 이벤트·텍스트. 단 **상장 은행만** 커버 |
| GDELT | 뉴스가 필요하면 유일한 무료·재현·귀속 가능 소스; 귀속 노이즈 큼 |

**Tier 3 — 제외**

| 소스 | 탈락 질문 |
|---|---|
| NewsAPI | (1) 1개월 lookback + 상업 사용 금지 |
| Alpha Vantage News | (1)(3) 데일리 쿼터·재현성 |
| Benzinga | (3) 유료 |
| Reuters via UCLA | (3) 접근 불확실·재배포 제약 |
| Company IR / Transcripts | (2) 표준 API 없음·수작업; 구조화 버전은 EDGAR가 대체 |
| yfinance | (3) 비공식 스크래핑(ToS 회색지대) + 상장 은행만 |
| FDIC/Fed/OCC Press RSS | 이미 구조화된 enforcement/failure로 대체 (한계 가치 낮음) |

## 공유 계약 (팀 간 접점은 이것뿐)

- `db/migrations/011_scoring_tables.sql`: `item_label` (3-클래스 `label`,
  `label_source`, `model_meta`, 아이템×라벨러당 한 행)과 `item_score`
  (3-클래스 `label`, `probs`, `keywords`, `model_version`, 아이템당 한 행),
  그리고 `raw_item(finbert_status = 'pending')` 부분 인덱스.
- `raw_item.finbert_status` 값 계약 (관례이며 CHECK 없음 — 기존 자유
  텍스트 사용과 일치):
  - `'pending'` (기본값, 수집이 설정) → 스코어링 대기 중
  - `'done'` → 스코어링 완료, `item_score`에 행 존재
  - `'failed'` → 스코어링 에러; `last_error`에 메시지
  - `'skipped'` → 의도적으로 스코어링 안 함 (예: 비영어); 사유는 `last_error`
- `label_source` 값: `'llama_kaggle' | 'gemini' | 'human'` (011의 CHECK;
  라벨러 추가는 `raw_item.source`와 같은 "ALTER 한 줄" 패턴).

## Stage 1 — 라벨링 (학습 코퍼스 만들기)

라벨은 **기사 텍스트만으로** 부여한다 — 이후 실제 사건과 대조해 검증하지
않는다 (멘토 지시; 그 비교는 fundamentals 테이블의 distress 라벨을 쓰는
백테스트의 몫이다).

### 대상 선정 (무엇을 라벨링하나)

export 시점에 적용하며, 저장하지 않는다. export dry-run은 6단계 퍼널을
보고한다 (멘토, 2026-07-17): **수집 총량 → 중복 제거 후 고유 → 영어 적격
→ 은행 관련 → 이벤트 관련 → 라벨링 최종 선정** — 각 단계의 건수를 보여
실제 작업셋 크기가 드러나게 한다.

1. 소스 간 중복 제거: 같은 기사가 GDELT, Alpha Vantage, NewsAPI, RSS,
   신디케이션 도메인을 통해 들어온다; `title_hash` dedup을 한 소스 안에서만이
   아니라 **소스 간에도** 적용한다. 정확한 규칙은 노이즈 predicate와 함께
   export 스크립트에서 확정한다.
2. 영어 아이템만 (2026-07-16 EDA 기준 코퍼스의 ~70%). FinBERT는
   영어 전용; 비영어 처리는 Phase-3의 질문이다.
3. 분석가/보유자 와이어 템플릿 제거 — 이것이 `is_syndication_noise`
   술어다 (13F/holdings 스팸, EDA 기준 GDELT의 ~14%). 2026-07-25에
   2026-07-22 배치로 실측: 제목 템플릿 매칭으로 **GDELT 7,756건 중
   1,578건(20.3%)**이 걸린다 — `"X Purchases N Shares of Y"`,
   `"X Has $N Million Stake in Y"`, `"X Raises Price Target for Y"`.

   이들은 귀속만 막을 게 아니라 **폐기**해야 한다. 그중 196건이 *방향성*
   Llama 라벨을 갖고 있는데, 그 방향은 다른 회사의 것이다("Goldman Sachs
   Raises AMD Price Target" → `positive`는 AMD의 것이지 골드만의 것이
   아니다). 이걸로 학습하면 모델은 "X가 Y의 목표가를 올림 = positive"를
   배우고, 그 패턴이 진짜 은행 행에서 발화해서 **귀속 게이트가 없애려는
   바로 그 오귀속을 스스로 만들어낸다**. 무해한 채움 데이터가 아니라
   유해하다.

   ⚠️ **술어 미확정.** 단순한 `Upgrad|Downgrad` 항은 은행 *자신의* 등급
   변동까지 잡는다("Commerce Bancshares Downgraded by Wall Street Zen to
   Sell" — 정당한 `negative`이며 gold slice에서 확인됨). 패턴은 **목적어로
   두 번째 엔티티**가 있을 때만 발화해야 한다. `n_duplicates`와 `domain`은
   여전히 유용한 신호다. export 스크립트에서 확정한다.
4. 텍스트 필드: GDELT 아이템은 `title` (본문은 수집하지 않음); EDGAR
   아이템은 `title` + `text_excerpt` (8-K 발췌, 최대 ~4000자).

EDA의 risk lexicon은 선정 기준이 **아니다** (결정 로그 2026-07-17 참조):
"bank explores strategic alternatives"나 "chief risk officer departs" 같은
기사는 lexicon 히트가 0이어도 리스크 관련성이 매우 높다. eligible한
아이템은 lexicon 매칭 여부와 무관하게 전부 라벨링한다.

**EDGAR 10-Q/10-K에는 스코어링할 텍스트가 없다.** `poll_edgar`는 **8-K에
대해서만** primary document 발췌를 가져오는데, 이는 올바른 선택이다: 10-Q의
primary document는 iXBRL이라 앞 4,000자가 산문이 아니라 택소노미 URI 덤프다
(2026-07-25에 7.6MB짜리 Goldman Sachs 10-Q로 확인 — 읽을 수 있는 텍스트는
약 180,000자 지점에서, MD&A는 약 654,000자 지점에서 시작). 그런데 EDGAR
title은 `"{holding_name} {form}"`으로 합성되어 **절대 비지 않기 때문에**
`empty_text` 가드가 걸리지 않고, 내용이 0인 107행(10-Q 103건, 10-K/A 4건)이
2026-07-22 라벨링 배치까지 들어갔다. 이 행들의 프롬프트 전문은
`Article: "Popular, Inc. 10-Q"`였고, Llama는 107건 전부를 `neutral`로 찍었다
— 퇴화 패턴이며 사람 라벨링 예산도 낭비됐다. `eligibility`는 EDGAR에 대해
excerpt가 비어있지 않을 것을 요구해야 한다. 분기 재무 상태는 이미 Call
Report fundamentals가 **숫자로** 제대로 커버하고 있으므로, 10-Q 산문에서
추출하는 것은 그것을 부정확하게 중복하는 일이다.

### Kaggle 왕복 계약 (수동 단계를 명시적으로)

1. **Export** (레포 스크립트, 추후): eligible한 `raw_item` 행을 SELECT →
   `labeling_batch_<date>.csv`, 컬럼은
   `raw_item_id, source, bank_id, published_at, title, text_excerpt`.
2. **Upload**: CSV를 비공개 Kaggle 데이터셋으로 업로드 (수동).
3. **Label** (Kaggle 노트북, Llama 8B급): 프롬프트가 행마다 정확히
   `positive|negative|neutral` 중 하나를 산출 → `labels_<date>.csv`, 컬럼은
   `raw_item_id, label, model_meta` (`model_meta` JSON: 모델 id, 양자화,
   프롬프트 버전, 실행 날짜).
4. **Import** (레포 스크립트, 추후): `label_source='llama_kaggle'`로
   `item_label`에 upsert; `ON CONFLICT (raw_item_id, label_source)` 업데이트
   — 새 프롬프트로 재라벨링하면 그 라벨러의 행을 덮어쓰고, 옛 프롬프트
   버전은 DB가 아니라 git의 `model_meta` 이력으로 남는다.

`raw_item`의 `llm_status` / `llm_attempts`는 라벨링 부기용으로 쓸 수 있으나
(예: `llm_status='labeled'`), 진실의 원천(source of truth)은 `item_label`이다;
어느 쪽이든 스키마 변경 없음 (README의 열린 선택지 유지).

### 수동 검증 & 품질 게이트

- 라벨링된 아이템 중 **200–300건을 무작위 샘플링** (클래스·소스별 층화);
  사람이 블라인드로 라벨링 (`item_label`에 `label_source='human'` 행).
- `llama_kaggle` 대비 클래스별 불일치를 측정. 무작위 노이즈는 허용 가능
  (파인튜닝은 노이즈에 강건함); **체계적 편향은 불가** (예: 규제 뉴스를
  전부 negative로 라벨링) — 리뷰는 패턴 찾기에 집중한다.
- **품질 게이트 (임계값은 팀이 결정, placeholder: 전체 일치 ≥85%, 모든
  클래스 75% 이상).** 게이트 실패 시: 프롬프트 수정 → 재라벨링 → 그래도
  실패하면 **Gemini challenger** 가동: 같은 코퍼스를 `label_source='gemini'`로
  라벨링하고, 수동 리뷰는 Llama/Gemini 불일치 행에 집중
  (`GROUP BY raw_item_id HAVING count(DISTINCT label) > 1`), 휴먼 라벨과의
  측정된 일치율로 팀이 champion을 선정한다.

## Stage 2 — 학습 (FinBERT 파인튜닝)

멘토에 따르면 pretrained FinBERT만으로는 불충분; stage-1 라벨로 파인튜닝한다.
라벨링과 같은 Kaggle GPU 환경 (워크플로 재사용).

- 학습셋: champion 라벨러의 `item_label` 행, 단 둘 다 존재하는 곳에서는
  `human` 행이 champion의 행을 덮어쓴다.
- 분할: 무작위가 아니라 **시간 기반** train/validation (예: 마지막 N주
  홀드아웃) — 서빙 현실과 일치하고, 신디케이션 준중복이 분할을 넘나들며
  누수되는 것을 막는다. 은행별 쏠림에 주의 (소수 은행이 GDELT 물량을
  지배함).
- 홀드아웃 슬라이스에서 accuracy + 클래스별 F1을 보고; pretrained-only
  FinBERT를 baseline으로 비교해 파인튜닝의 효과를 입증한다.
- 아티팩트: 모델 가중치는 레포 밖에서 버전 관리 (Kaggle 데이터셋 vs HF hub,
  TBD); 각 행을 어떤 가중치가 스코어링했는지는 `item_score.model_version`에
  기록된다.

### 학습셋 위생 (파이프라인이 아니라 학습셋 구성 시점에 적용)

Stage 2는 Stage 1에 걸려 있다: 학습셋은 **champion** 라벨러의 행이고,
champion은 품질 게이트가 정한다. 게이트 실패는 프롬프트 수정 → 재라벨링을
뜻하고 그러면 라벨이 전부 바뀌므로, 게이트가 결론 나기 전에 학습하면 그
학습을 버리게 될 위험이 있다.

게이트가 결론 나면, 일부 행은 학습셋에서 제외해야 한다. 이는 **학습셋을
조립하는 지점의 필터**이며 `eligibility` 변경도, 재export도, 재라벨링도
필요 없다:

| 부류 | 행수 | 방향 라벨 | 조치 |
|---|---|---|---|
| 무내용 EDGAR (10-Q / 10-K/A, excerpt 없음) | 107 | 0 | 제외 — 퇴화 패턴, 전부 `neutral` |
| 지분/13F 와이어 스팸 ("X Purchases N Shares of Y") | 723 | 51 | 제외 — 방향이 다른 회사 것 |
| 등급 / 목표주가 템플릿 | 375 | 123 | **일단 유지** — 아래 참조 |

세 번째 부류는 아직 안전하게 버릴 수 없다. 같은 표면형이 정반대 역할을
동시에 담고 있기 때문이다:

    "Pearson downgraded by JP Morgan"          bank_id=jpm   → jpm이 하는 쪽  → 버림
    "Commerce Bancshares Downgraded by WSZ"    bank_id=cbsh  → cbsh가 당하는 쪽 → 남김

이 부류를 통째로 버리면 진짜 은행 등급 변동을 포함한 방향 라벨 123건이
날아간다. 귀속된 은행이 *주어*인지 *행위자*인지 먼저 구분하는 규칙이
필요하다 (Stage 1의 3번 참조).

⚠️ **이 필터는 Stage 3 서빙이 나가기 전에 반드시 `eligibility`로 옮겨져야
한다.** 학습에서만 걸러내고 서빙은 그 행들을 계속 스코어링하면, 공유
eligibility 설계가 막으려는 바로 그 train/serve skew가 생긴다.

**알려진 리스크 — tone prior vs risk-direction.** FinBERT는 금융 *tone*으로
프리트레인됐지만, 우리 타겟은 *리스크 방향*이다. 둘은 라벨링 가이드가 가장
가치있다고 표시한 완곡어법 케이스에서 정확히 갈린다 — "exploring strategic
alternatives"(차분한 톤, negative), "consent order lifted"("regulator" 표현,
positive). FinBERT는 stage-1 Llama 라벨의 distillation이라 이 케이스들에서
Llama를 넘지 못한다. 따라서 품질 게이트(§ 수동 검증)는 전체만이 아니라
**tone≠direction 하위그룹으로 층화**해서 읽어야 한다: 전체 일치율은 통과해도
고가치 완곡어법 케이스가 실패할 수 있다. `bank_id` 부재(결정 로그 2026-07-24
참조)가 아니라 **이것**이 FinBERT 단계의 주요 정확성 리스크다.

## Stage 3 — 서빙 (일 단위 배치)

기존 스케줄 인프라에서 실행 (GitHub Actions, CPU 전용 — FinBERT급 모델은
CPU로 충분; 서빙이 LLM이 아닌 이유이며, QnA 논의 참조).

1. `raw_item`에서 `finbert_status='pending'`인 행 SELECT (부분 인덱스).
2. 라벨링과 **같은** eligibility 필터 적용; 부적격 → `'skipped'`.
3. 배치로 스코어링 → `item_score` upsert (`label`, `probs`, `model_version`).
4. `finbert_status='done'` 마킹 (또는 `'failed'` + `last_error`); 두 쓰기는
   배치당 하나의 트랜잭션으로 묶는다.
5. 기존 `write_heartbeat`(`pipeline/db.py`)로 heartbeat — poller들과 동일.

**키워드 / explainability (v2):** 감성 클러스터별 두드러진 키워드
(PCA / 키워드 클러스터링)가 이후 이터레이션에서 `item_score.keywords`를
채운다; 대시보드 계약이 안정되도록 컬럼은 지금 만들어 둔다.

## 은행 귀속 — 점수가 매겨졌는데도 그 은행에 반영되지 않는 이유

### 문제, 구체적으로

GDELT는 쿼리를 **기사 본문 전체**에 매칭하지만, `artlist` API는 **제목만**
돌려준다 (2026-07-25 확인: article 객체가 담고 있는 필드는 `domain,
language, seendate, socialimage, sourcecountry, title, url, url_mobile`이
전부 — 본문도, 발췌도, 매치된 스니펫도 없다). 그래서 `gs` 폴더에 이런 행이
들어온다:

    bank_id : gs
    title   : "Chipmaker SK Hynix raises $26.5bn in US stock market debut"

매치 자체는 정당하다 — 본문에 "Goldman Sachs가 주관사"라고 적혀 있었다.
하지만 제목의 주인공은 SK하이닉스다. 이 제목을 `positive`로 채점해서 `gs`에
반영하면, 골드만은 벌지 않은 호재 점수를 얻는다.

드문 예외가 아니다:

| 실측 (2026-07-22 배치 / gold slice) | |
|---|---|
| 제목에 자기 `bank_id` 은행명이 있는 GDELT 행 | **726 / 7,756 (9.4%)** |
| 방향성 행(non-neutral) 중 엉뚱한 은행에 귀속된 것 | **21 / 30 (70%)** |

neutral 노이즈는 무해하다 — 어느 쪽이든 방향이 0이다. 하지만 **방향성 있는
행의 오귀속은** 실제 방향을 엉뚱한 은행에 밀어 넣는다.

### 왜 `eligibility` 필터가 아닌가

`eligibility`는 공유 **텍스트** 게이트다: "여기 채점할 만한 게 있나?"
귀속은 다른 질문에 답한다: "이 점수가 *이* 은행 것이 맞나?" 둘을 합치고
싶어지지만 비싸다:

- 제목에 은행명이 있을 것을 요구하면 라벨링을 하기도 전에 **GDELT 코퍼스의
  ~90%가 날아간다.**
- 그 행들은 *유효한* 학습 데이터다. "Bandhan Bank Q1 results beat estimates,
  but stock crashes"는 교과서적인 은행 악재 헤드라인이고 `negative` 라벨도
  정확하다 — 다만 JPMorgan의 `negative`가 아닐 뿐이다.

그래서 두 축을 분리해서 둔다:

    라벨링 / 학습      eligible한 아이템 전부   → 코퍼스 보존
    스코어링           eligible한 아이템 전부   → 당분간만, 아래 참조
    은행별 집계        귀속 가능한 것만         → 가짜 신호 제거

**귀속 실패 행을 채점하는 것은 한시적 단계이지 영구 규칙이 아니다.** 행은
은행에 묶여 있다 — `raw_item`의 UNIQUE가 `(source, external_id, bank_id)`라
두 은행의 쿼리에 걸린 기사는 두 개의 행이 된다 — 그리고 한 행의 점수는
오직 그 행의 은행 롤업에서만 읽힌다. 귀속에 실패한 순간, 그 점수를 읽는
주체는 아무도 없다.

그럼에도 *게이트가 어린 지금은* 계산할 가치가 있다. 게이트는 6행으로
검증됐다. "이게 진짜 신호를 버리고 있나?"를 물으려면 **버려진 쪽의 점수**가
있어야 한다. 채점을 건너뛰면 버린 더미를 읽을 수 없고, 게이트를 넓히거나
좁힐 근거 자체가 없어진다.

    게이트 미검증 (지금)   전부 채점                  → 게이트를 관측 가능하게 유지
    게이트 안정화 후       finbert_status='skipped'   → 낭비 중단

비용은 어느 쪽이든 결정 요인이 아니다 — FinBERT는 CPU급이고 일일 증분은
작다; 7,756행은 일회성 백로그 수치다. 전환도 양방향으로 싸다: `'skipped'`는
이미 상태 계약에 있고, `'pending'`으로 되돌리면 일일 배치가 다시 주워간다.

### 체크

**제목에 자기 은행명이 있나?** `bank.aliases`(이미 시드에 있음)로 단어경계
매칭. `\b`가 필수다 — 안 그러면 bare `"Citi"`가 "Citizens"에 걸린다. 실패한
행도 라벨링·스코어링은 그대로 하고, **해당 은행 롤업에서만** 제외한다.

귀속 체크는 이것이 전부다. 분석가/보유자 와이어 템플릿("Goldman Sachs Raises
AMD Price Target")은 여기서 다루지 **않는다** — 방향성 라벨이 구조적으로
다른 회사의 것이라 학습을 오염시키므로, 더 앞단인 Stage 1 대상 선정 3번에서
폐기된다. 두 술어는 *처리 방식*으로 의도적으로 나뉜다:

    학습에 유해              → Stage 1에서 폐기      (와이어 템플릿)
    학습엔 유효, 배달만 잘못  → 여기서 귀속만 제외    (제목에 은행명 없음)

`gold_slice_3`(GDELT 42행 — 행별 근거가 달린 슬라이스)으로 검증: 오귀속
방향성 행 3건 전부 차단, 정상 귀속 방향성 행 3건 전부 통과. 분석가 템플릿
1건이 이 체크를 통과했는데, 그것이 바로 그 부류를 Stage 1에서 처리하는
이유다. n이 작으므로 슬라이스가 더 라벨링되면 재측정한다.

## 백테스트 연계 (scoring 범위 밖, 방향 제시용)

감성 점수와 fundamentals의 조인:
`bank.fdic_cert ↔ fact_bank_quarter.fdic_cert_number` (FK 없음, 값으로 조인).
Distress 라벨: `fact_distress_event`, `fact_bank_quarter.distress_within_4q/8q`.
EDA에서 온 주의사항: 실패 이벤트 ~3,627건 중 GDELT 시대(2017+)에 속하는 건
~27건뿐이고, 4분기 양성은 은행-분기의 ~0.45% — 백테스트는 희소 이벤트
평가이며 그에 맞게 설계되어야 한다.

## 미결 항목 (팀 결정 사항 — README: "teammates own all design decisions")

- [ ] scoring/ 소유자 (현재 TBD)
- [ ] 품질 게이트 임계값 (위의 placeholder)
- [ ] 대상 선정용 신디케이션 노이즈 predicate 확정
- [ ] 모델 가중치 호스팅 (Kaggle 데이터셋 vs HF hub)
- [ ] `probs`를 jsonb로 할지 numeric 컬럼 3개로 할지 (일단 jsonb 선택;
      대시보드가 확률로 직접 정렬/필터해야 하면 재검토)
- [ ] 추론 시점 LLM 에스컬레이션 티어(저신뢰 아이템 → LLM)를 추진할지 —
      `llm_status`/`llm_attempts`는 그 용도로 미사용 상태 유지
