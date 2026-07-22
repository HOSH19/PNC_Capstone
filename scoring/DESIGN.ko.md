# scoring — 설계 (Phase 2)

> 영어 원본: [DESIGN.md](DESIGN.md) — 두 문서가 어긋나면 영어본이 기준.

[README.md](README.md)의 자매 문서. 2026-07-12 멘토와 합의한 방법론:
**LLM 보조 라벨링 → BERT 계열 모델 파인튜닝 → 일 단위 배치 예측**, 키워드
수준 explainability 포함. 이 문서는 구체적인 계약(contract)을 고정한다;
구현은 팀 리뷰 이후에 시작한다.

## 결정 로그

| 날짜 | 결정 | 근거 |
|---|---|---|
| 2026-07-15 | 라벨러: **Kaggle 무료 GPU의 Llama, 처음엔 단독**. Gemini API는 비상용 challenger로, 초기에는 셋업하지 않음. | Kaggle 환경은 FinBERT 파인튜닝(stage 2)에 재사용됨; API 비용/ToS/rate-limit 의존성 없음. Gemini 투입 조건은 "품질 게이트" 참조. |
| 2026-07-15 | 저장소: **새 테이블 2개** (`item_label`, `item_score`), `raw_item`은 무변경. | 라벨링 출력은 (아이템, 라벨러)별 이력(champion-challenger + 휴먼 행이 공존해야 함); 서빙 출력은 아이템당 최신 점수 하나. 수명주기가 다름 → 테이블 분리. 마이그레이션: `db/migrations/011_scoring_tables.sql`. |
| 2026-07-15 | 이 문서의 범위: 설계 + 스키마 계약만, 실행 코드는 아직 없음. | 팀원들의 데이터 소스 연동과 scoring 설계가 병렬로 진행될 수 있게 함. |
| 2026-07-17 | Risk lexicon은 **서술용 지표일 뿐 — 라벨링 eligibility 필터로 절대 사용하지 않음**. eligibility = 언어 + 노이즈 필터, 그 외 없음. | 멘토 피드백: 중요한 은행 리스크 기사에는 명백한 부정 단어가 없는 경우가 많음 ("bank explores strategic alternatives", "deposits fall for third consecutive quarter", "chief risk officer departs"); lexicon 필터는 바로 그런 기사들을 걸러내 버림. |
| 2026-07-17 | 코퍼스 집계는 **6단계 퍼널**을 사용; export dry-run이 6개 카운트를 모두 보고해야 함. | 멘토 피드백: 총량만 보면 소스 간 중복이 가려짐 (같은 기사가 GDELT / Alpha Vantage / NewsAPI / RSS / 신디케이션 도메인으로 유입); 퍼널이 실제 작업셋 크기를 드러냄. 단계는 아래 "대상 선정" 참조. |
| 2026-07-21 | 데이터 소스를 **3개 질문(재현성·귀속·지속성)**으로 티어링; NewsAPI / Alpha Vantage / Benzinga / Reuters(UCLA) / Company IR / yfinance는 **제외**. | 아래 "데이터 소스 선정 기준" 참조. 뉴스API 제외의 결정적 근거는 쿼터가 아니라 **과거 복원 불가**(NewsAPI 무료 lookback ~1개월 → 백테스트 학습 불가). |

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
3. 13F/holdings 신디케이션 노이즈 제거 (EDA 기준 GDELT의 ~14%):
   `title_hash` 클러스터가 holdings 스팸인 행을 제외; `n_duplicates`와
   `domain`이 식별에 도움이 된다. 정확한 predicate는 export 스크립트에서
   확정한다.
4. 텍스트 필드: GDELT 아이템은 `title` (본문은 수집하지 않음); EDGAR
   아이템은 `title` + `text_excerpt` (8-K 발췌, 최대 ~4000자).

EDA의 risk lexicon은 선정 기준이 **아니다** (결정 로그 2026-07-17 참조):
"bank explores strategic alternatives"나 "chief risk officer departs" 같은
기사는 lexicon 히트가 0이어도 리스크 관련성이 매우 높다. eligible한
아이템은 lexicon 매칭 여부와 무관하게 전부 라벨링한다.

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
