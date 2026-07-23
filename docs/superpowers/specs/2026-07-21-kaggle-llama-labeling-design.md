# Kaggle Llama 라벨링 셋업 설계 스펙

- **날짜**: 2026-07-21
- **상태**: 핵심 결정 확정 (런타임 vLLM, 리스크 방향 라벨, few-shot) — 검토 대기
- **관련 문서**: [scoring/DESIGN.md](../../../scoring/DESIGN.md) Stage 1 "Kaggle 왕복 계약", [scoring/INTEGRATION.md](../../../scoring/INTEGRATION.md) Pattern A

## 배경 / 문제

FinBERT 파인튜닝(Stage 2)에 쓸 **학습 라벨이 없다**. 사람이 수만 건을 다 달
수 없으므로, Kaggle 무료 GPU에서 Llama 8B급 모델로 각 기사에 감성 라벨
(`positive|negative|neutral`)을 대량 생성한다(weak supervision). 이 스펙은
그 **라벨링 노트북 + 프롬프트 + 런타임**을 정의한다.

이 작업은 이미 고정된 계약을 소비/생산한다:
- **입력**: export CSV `labeling_batch_<date>.csv`
  (`raw_item_id, source, bank_id, published_at, title, text_excerpt`).
  export 대상은 `pipeline/eligibility.py`가 판정.
- **출력**: `labels_<date>.csv` (`raw_item_id, label, model_meta`).
- **프롬프트 버전**은 DB가 아니라 `evals/prompts/<이름>_llama.md`(git).

## 목표

- Kaggle 노트북 하나로: export CSV 읽기 → 배치 추론 → 출력 검증 →
  `labels_<date>.csv` + `model_meta` 기록.
- 행마다 **정확히 하나의 유효 라벨**을 보장.
- 재현 가능한 디코딩(그리디/temperature=0)으로 같은 입력 → 같은 라벨.

## 비목표

- export/import 스크립트(레포 측, 별도 작업).
- Gemini challenger(품질 게이트 실패 시 별도).
- 파인튜닝(Stage 2).

---

## 결정 1: 런타임 = vLLM (선택지 A)

수만 건의 **짧고 길이가 제각각인** 프롬프트를 Kaggle 무료 GPU(주간 시간
쿼터 존재)에서 돌려야 한다. 이 워크로드에서 병목은 **처리량(throughput)**이고,
그게 곧 GPU 쿼터 소모로 직결된다. 세 후보를 처리량 관점에서 비교:

### A. vLLM — 채택

두 가지 핵심 기술이 배치 라벨링에 정확히 맞는다:

1. **Continuous batching (in-flight batching)**
   순진한 배치는 "한 배치의 모든 문장이 끝날 때까지" 기다린다 — 가장 긴
   시퀀스가 배치 전체를 붙잡는다(head-of-line blocking). 우리 프롬프트는
   길이 편차가 크고(제목만 vs 8-K 발췌 4000자) 출력은 한 단어라, 이 낭비가
   특히 심하다. vLLM은 한 시퀀스가 끝나 슬롯이 비는 즉시 대기 중인 다음
   요청을 밀어넣어 GPU를 계속 채운다.
2. **PagedAttention (KV-cache 페이징)**
   KV 캐시를 OS 가상메모리처럼 페이지 단위로 관리해 단편화를 없앤다. 같은
   VRAM으로 **훨씬 큰 배치**를 올릴 수 있어 처리량이 올라간다.

부수 이점: **guided decoding**(`guided_choice=[...]`)으로 출력을
`{positive, negative, neutral}` 세 값 중 하나로 **디코딩 단계에서 강제**할 수
있다 → 파싱 실패/잘못된 출력 자체가 원천 차단(아래 결정 3).

결과적으로 순진한 배치 대비 대개 10× 이상 처리량. 수만 건이 1~2시간 vs
10시간+로 갈리고, 이는 Kaggle 주간 GPU 쿼터 안에 드느냐를 좌우한다.

### B. transformers + 4bit(bitsandbytes) — 미채택

`model.generate()` 루프 + 정적 배치. bitsandbytes 4bit는 **VRAM은 아끼지만**
스텝마다 dequantize 오버헤드가 있어 토큰당 속도는 오히려 느려질 수 있고,
근본 병목인 정적 배치/HOL blocking은 그대로다. 수백 건이면 충분하지만
수만 건 배치엔 부적합. (셋업은 A보다 간단 — 소량 프로토타이핑엔 유용.)

### C. Ollama — 미채택

로컬/단일 사용자 대화형 서빙에 최적화된 도구다. 고처리량 오프라인 배치용
continuous batching이 없고 병렬성이 제한적이며, Kaggle의 휘발성 환경에서
서버 프로세스를 띄우는 것도 어색하다. 개발 노트북엔 최고지만 배치 라벨링엔
잘못된 도구.

**요약**: 병목=처리량=쿼터. continuous batching + PagedAttention을 갖춘
**A(vLLM)**가 이 워크로드에 정확히 맞는다.

## 결정 2: 모델 & 양자화

- **모델**: Llama 3.1 8B Instruct (지시 튜닝판 — 라벨 지시 준수에 필요).
- **양자화**: **AWQ 4bit** (vLLM이 네이티브 지원). fp16 8B ≈ 16GB로 Kaggle
  단일 16GB 카드(T4/P100)엔 빠듯한데, AWQ 4bit ≈ 5~6GB로 여유롭게 올라가
  배치를 더 키울 수 있다.
- **대안(검증 노브)**: AWQ 체크포인트가 여의치 않으면 GPTQ 4bit, 또는
  T4×2 텐서 병렬로 fp16. 최종 값은 실제 Kaggle 환경에서 확정.

## 결정 3: 출력 유효성 = guided decoding으로 보장

- vLLM `SamplingParams(guided_choice=["positive","negative","neutral"], temperature=0)`.
- 모델이 세 값 외를 낼 수 없으므로 파싱/재시도 로직이 사실상 불필요.
- 방어적 검증: 반환값이 세 집합에 있는지 assert(guided decoding 미사용 대안
  경로 대비). 위반 행은 조용히 넘기지 말고 카운트+로깅.

## 결정 4: 라벨의 의미 = "리스크 방향" (순수 감성 아님)

라벨 값은 `positive/negative/neutral`로 같지만, **무엇을 재느냐**를 확정해야
한다. 두 가지 프레임이 있다:

| 프레임 | negative의 의미 | 재는 대상 |
|---|---|---|
| 순수 감성(tone) | 기사의 **말투가 부정적** | 글의 감정 톤 |
| **리스크 방향(채택)** | 이 기사가 은행 **리스크 상승**을 시사 | 우리가 원하는 신호 |

대부분은 둘이 일치하지만, **어긋나는 경우가 제일 중요한 케이스**다:

| 기사 | 순수 감성 | 리스크 방향 |
|---|---|---|
| "bank explores strategic alternatives" | neutral(담담) | **negative**(매각·부실 전조) |
| "regulator lifts consent order" | negative처럼('규제') | **positive**(리스크 하락) |

**리스크 방향을 택한 이유**:
1. 우리 목표(부실 조기경보)와 **직접 정렬** — 라벨이 목표의 대리가 아니라 목표 그 자체.
2. 멘토가 명시한 우려(중요 리스크 기사엔 명시적 부정어가 없는 경우가 많음)를
   잡는 **유일한** 길. 순수 톤은 "explores strategic alternatives"류를 놓친다.
3. Risk lexicon을 필터로 쓰지 않기로 한 결정(DESIGN 2026-07-17)과 논리적으로 일관.

**비용과 상쇄**: 리스크 방향은 라벨 판단이 더 어렵고 사람마다 갈릴 수 있다
(노이즈↑). 이를 (a) 또렷한 정의와 (b) few-shot 예시(아래)와 (c) 사람 품질
게이트로 상쇄한다.

## 라벨링 프롬프트 (확정 초안)

`evals/prompts/<이름>_llama.md`에 저장, 버전은 파일로 관리. 라벨은 **기사
텍스트만으로** 부여(결과 훔쳐보기 금지 — 미래 정보 leakage 방지). 코퍼스가
영어 전용이므로 **프롬프트·예시는 영어**(입력 분포와 정렬).

### 3-클래스 정의 (리스크 방향)

- **negative** — 은행의 리스크 상승/건전성 악화 시사: 손실, 예금 이탈, 규제
  조치, 소송, 경영진(특히 리스크·재무) 이탈, "전략적 대안 모색" 등 완곡한
  distress 신호 포함.
- **positive** — 리스크 하락/건전성 개선 시사: 자본 확충, 규제 조치 해제,
  실적 개선, 신용등급 상향 등.
- **neutral** — 명확한 리스크 방향 없음: 일상적 발표, 상품 출시, 지점 개설,
  스폰서십, 은행이 부수적으로만 언급된 기사 등.

### few-shot 예시 (프롬프트에 포함)

경계 케이스(특히 **완곡한 리스크**)를 예시로 각인해 순수 톤으로 새는 것을
막는다.

```
Article: "Regional Bank reports third straight quarter of deposit outflows"
Label: negative

Article: "Community Bancorp says it is exploring strategic alternatives"
Label: negative        # euphemistic distress signal — tone is calm, risk is up

Article: "Pinnacle Bank's chief risk officer resigns after two years"
Label: negative        # risk/finance leadership departure

Article: "Federal Reserve lifts consent order against Midwest Bank"
Label: positive        # sounds regulatory, but risk is going DOWN

Article: "Summit Bank raises $500M in capital, lifting its Tier 1 ratio"
Label: positive

Article: "Coastal Bank opens three new branches in the metro area"
Label: neutral
```

### 프롬프트 템플릿 (Llama가 보는 것)

```
You label news articles about US banks by the RISK DIRECTION they imply
for the bank — not by the article's emotional tone.

- negative: implies the bank's risk is RISING / health worsening (losses,
  deposit outflows, enforcement, lawsuits, risk/finance executive exits,
  "exploring strategic alternatives", and other euphemistic distress signals).
- positive: implies risk FALLING / health improving (capital raises,
  consent orders lifted, earnings improvement, rating upgrades).
- neutral: no clear risk direction (routine announcements, product launches,
  branch openings, sponsorships, incidental mentions).

Examples:
{few-shot examples above}

Now label this article. Answer with exactly one word: positive, negative,
or neutral.

Article: "{title / title + excerpt}"
Label:
```

출력 유효성은 결정 3의 guided decoding(`guided_choice`)으로 강제 — 위
"one word" 지시는 보조.

## 노트북 흐름

1. 입력 CSV를 Kaggle 데이터셋 경로에서 로드.
2. 각 행의 텍스트 조립: `source`별 규칙(gdelt=title, edgar=title+excerpt).
   — eligibility의 `text_fields`와 동일 규칙(중복 정의 아님, 개념 일치).
3. 프롬프트 렌더링 → vLLM 배치 추론(guided_choice, temperature=0).
4. `raw_item_id, label` 수집.
5. `model_meta` = {model id, quantization, prompt_version, run_date} — 실행 날짜는
   노트북 파라미터로 주입(재현성).
6. `labels_<date>.csv` 저장 → Kaggle output에서 다운로드.

## model_meta 스키마

```json
{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "quantization": "awq-4bit",
  "prompt_version": "v1",
  "run_date": "2026-07-21"
}
```

## 수동 단계 (사용자)

1. Kaggle 노트북 생성, **Accelerator = GPU(T4×2 또는 P100)** 켜기.
2. export CSV를 **비공개 Kaggle 데이터셋**으로 업로드, 노트북에 첨부.
3. vLLM/모델 설치 셀 실행(첫 실행은 가중치 다운로드로 수 분).
4. 라벨링 셀 실행 → `labels_<date>.csv` 다운로드 → 레포 import 스크립트로.

## 검증 (Definition of done)

- 소규모(예: 20행) 샘플로 노트북 end-to-end 1회: 모든 행에 유효 라벨,
  `labels_<date>.csv` 스키마 일치, `model_meta` 채워짐.
- temperature=0 재실행 시 라벨 동일(재현성 확인).
- 클래스 분포가 상식적인지 눈으로(전부 neutral 등 이상 신호 없나).

## 열린 질문 (검토)

- AWQ 체크포인트 소스(공개 리포) 확정 — 실제 Kaggle 환경에서 검증.
- few-shot 예시를 실제 코퍼스 라벨링 후 오분류 패턴을 보고 보강할지(반복 개선).
