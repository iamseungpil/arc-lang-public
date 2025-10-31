# GPT-OSS 20B ARC 실험 보고서 (2025-10-31)

## arc-lang 모델 구조

arc-lang 시스템은 RunConfig 기반의 multi-step 아키텍처를 통해 추상적 추론 문제를 해결한다. 시스템은 instruction model과 follow model로 구성되며, instruction model은 training examples를 분석하여 자연어 규칙을 생성하고, follow model은 그 규칙을 읽고 test input에 대한 grid를 생성한다. 현재 실험에서는 두 역할 모두 GPT-OSS 20B가 수행한다.

각 문제는 3단계의 instruction step과 1단계의 revision pool을 거치며, 각 단계는 설정된 횟수만큼 반복 실행된다. 첫 번째와 두 번째 step은 각각 2회, 세 번째 step은 3회 반복되어 총 7개의 instruction 후보를 생성한다. Revision pool은 상위 3개 instructions를 선택하여 각각 2회 수정을 시도하며, 최종적으로 3회의 follow-up 생성을 통해 답안을 산출한다. Instructions의 품질은 leave-one-out cross-validation으로 평가되며, 각 training example을 test로 사용했을 때의 pixelwise accuracy 평균으로 계산된다.

시스템은 GPT-OSS 20B 모델을 vLLM 서버를 통해 호출하며, GPU 0과 1을 사용한 tensor parallelism으로 추론을 수행한다. vLLM은 OpenAI-compatible API를 제공하며, progressive token scaling 전략을 통해 8,000 토큰에서 시작하여 응답이 잘릴 경우 12,000, 16,000, 20,000 토큰으로 점진적으로 한도를 증가시킨다. 이 방식은 GPT-OSS 20B의 reasoning mode에서 생성되는 긴 자연어 추론 과정을 완전히 캡처하는 데 필수적이다. 서버는 자동 큐잉 시스템을 통해 여러 요청을 순차 처리하며, max_concurrency를 4로 설정하여 동시에 처리할 수 있는 작업의 수를 제한한다.

각 단계에서 모델은 JSON structured output을 생성하도록 구성되며, response_format으로 json_object 타입을 지정한다. temperature는 0.3으로 설정되어 비교적 결정론적인 출력을 유도하지만, GPT-OSS 20B의 reasoning mode는 여전히 활성화되어 모델이 문제 해결 과정에서 자연어 추론을 먼저 수행한 후 JSON 답변을 생성한다. 이 구조는 단순히 하나의 답변을 생성하는 것이 아니라, 여러 단계에 걸쳐 점진적으로 문제를 분석하고 수정하는 iterative refinement 방식을 구현한다.

## Hyperparameter 설정

본 실험의 hyperparameter는 로컬 GPU 환경(2×A100)에 맞게 설정되었다. 2025년 10월 29일 커밋(86dd8d6)에서 추가된 `local_gpt_oss_20b_config`는 메모리 제약과 처리 속도를 고려하여 보수적으로 구성되었다.

| Parameter | 값 | 설명 |
|-----------|---:|------|
| **Step 1 times** | 2 | 첫 번째 instruction 생성 횟수 |
| **Step 2 times** | 2 | 두 번째 instruction 생성 횟수 |
| **Step 3 times** | 3 | 세 번째 instruction 생성 횟수 |
| **RevisionPool top_scores** | 3 | 수정할 상위 instructions 개수 |
| **RevisionPool times** | 2 | 각 instruction당 수정 시도 횟수 |
| **final_follow_times** | 3 | 최종 답안 생성 횟수 |
| **timeout_secs** | 300 | 각 단계별 timeout (5분) |
| **max_concurrent_tasks** | 4 | 동시 처리 작업 수 |
| **temperature** | 0.3 | 출력 결정론성 제어 |
| **include_base64** | False | 이미지 base64 인코딩 사용 안 함 |
| **use_diffs** | True | Grid diff 사용 |

이 설정은 총 7개의 초기 instructions (2+2+3)를 생성하고, 상위 3개를 각각 2회 수정하며, 최종적으로 3개의 답안 후보를 생성한다. GPT-5 Pro 설정(Step 3: 20회, timeout: 3시간)에 비해 샘플링 횟수가 6.7배 적고 timeout이 36배 짧지만, 로컬 하드웨어에서 안정적으로 실행 가능하도록 조정된 값이다. 더 많은 샘플링을 통해 성능 개선 여지가 있으나, 현재 설정에서도 65.75%의 정확도를 달성했다.

## 실험 결과

| 항목 | 수치 |
|------|------|
| 완료 문제 수 | 146/400 (36.50%) |
| 정확도 | 65.75% (96/146 correct) |
| 실험 시간 | 46시간 이상 |
| 모델 | GPT-OSS 20B (openai/gpt-oss-20b) |
| 하드웨어 | GPU 0,1 (vLLM tensor parallelism) |

실험은 ARC evaluation 세트 400문제 중 146문제를 완료했으며, 이 중 96문제를 올바르게 해결하여 65.75%의 정확도를 기록했다. 평균적으로 각 문제는 약 19분 소요되었으며, 이는 arc-lang의 multi-step 구조와 GPT-OSS 20B의 reasoning mode로 인한 긴 응답 생성 시간에 기인한다. 시스템은 메인 실험과 병렬로 1개 문제(695367ec)에 대한 content 로깅 실험을 수행했으며, vLLM의 자동 큐잉을 통해 두 실험 간 간섭 없이 안정적으로 실행되었다. content 로깅 실험은 10개의 응답 블록을 캡처했으며, 이를 통해 모델이 문제를 해결하는 실제 추론 과정을 완전히 분석할 수 있었다.

## Solved 문제 분석: 695367ec

Task 695367ec는 작은 입력 그리드(1×1, 2×2, 3×3 등)를 15×15 출력 그리드로 확장하며, 입력 색상으로 규칙적인 격자 패턴을 생성하는 문제다. content 로깅을 통해 캡처한 GPT-OSS 20B의 실제 추론 과정을 보면, 모델은 두 가지 해석을 시도했다.

첫 번째 해석에서 모델은 "Create a 15×15 grid. For every row and column whose index is a multiple of 4 (i.e., the 4th, 8th, and 12th rows and columns), fill that entire row or column with the single value that appears in the input grid. All other cells are filled with 0"라는 규칙을 제시했다. 이는 4의 배수 인덱스(3, 7, 11)에 수평선과 수직선을 그어 십자 격자를 만드는 방식이다.

두 번째 해석은 더 정교한 알고리즘적 접근으로, "1. Determine the side length n of the input square grid. 2. Let s = n + 1. 3. Compute how many equally spaced lines can fit in a 15×15 grid: m = floor(15 / s). 4. Create a 15×15 grid filled with 0. 5. For each integer k from 1 to m, compute the index p = k * s – 1 (this is 0-based). 6. Set every cell in row p to the input value. 7. Set every cell in column p to the input value"라는 단계별 절차를 명시했다. 이 방식은 입력 크기에 따라 간격을 동적으로 계산하여 일반화된 해법을 제공한다.

모델은 훈련 예제에서 입력 크기가 다양함에도 출력이 항상 15×15임을 파악하고, 입력 크기와 격자 간격의 관계(s = n + 1)를 정확히 추출했다. 성공 요인은 명확한 수학적 패턴, 일관된 출력 크기, 그리고 색상값의 의미가 단순히 "채우기"로만 작용하는 구조에 있다. 모델은 복잡한 의미론적 해석 없이 순수한 규칙 기반 grid operation으로 문제를 해결했으며, 이는 GPT-OSS 20B가 명시적 패턴 인식과 알고리즘적 추론에 강점을 보임을 시사한다.

## Unsolved 문제 분석: 7bb29440

![Task 7bb29440 (Unsolved)](/home/ubuntu/arc_agi_jeremy/analysis/images/incorrect_7bb29440.png)

Task 7bb29440는 모델이 실패한 대표적인 사례로, 4×4 입력을 16×16 출력으로 확장하는 문제다. 모델은 이 문제를 "가장 빈번한 색상(5)을 찾아 해당 색상이 있는 위치에 입력 그리드 전체를 복사하고 나머지는 0으로 채우는" block-based replication으로 해석했다. reasoning 로그에서 모델은 색상 5가 6회 등장함을 정확히 계산하고, (0,1), (1,0), (1,1), (2,2), (2,3), (3,2) 위치에 4×4 입력 그리드를 배치하는 16×16 출력을 생성하려 시도했다. 그러나 실제 정답은 각 색상 cell이 특정 decorative cross motif로 확장되는 template instantiation 패턴이었으며, 단순 grid replication이 아닌 색상별 고유한 장식 패턴 생성을 요구했다. 모델은 high-level 구조(small input → large output with pattern repetition)는 파악했으나, 색상이 abstract symbol로 작용하여 구체적인 decorative template을 호출한다는 semantic abstraction을 이해하지 못했다. 이는 모델이 명시적 geometric transformation에는 강하지만, implicit template structure와 multi-level abstraction이 필요한 문제에서는 취약함을 보여준다.

## 기술적 세부사항

본 실험은 content 로깅을 위해 `src/llms/structured.py`의 `_get_next_structure_local_vllm` 함수를 수정하여 각 vLLM 응답의 전체 content를 `[FULL_RESPONSE_CONTENT_START/END]` 태그로 감싸 로그에 기록했다. GPT-OSS 20B는 OpenAI의 별도 reasoning 필드를 사용하지 않고 모든 reasoning을 content 필드에 자연어로 포함시키기 때문에, 원본 메인 실험 로그에서는 JSON 추출 후 자연어 설명이 모두 폐기되었다. 이번 content 로깅 수정을 통해 비로소 모델이 생성한 "instructions" 필드의 자연어 추론 과정을 완전히 캡처할 수 있었다.

실험은 `task_ids={"695367ec"}`로 1개 solved 문제만 실행하도록 `src/run.py`를 수정했으며, 저장 경로를 `/data/arclang/attempts/content_logging_solved/`로 완전히 분리하여 메인 실험 데이터 덮어쓰기를 방지했다. `LOG_LEVEL=DEBUG`와 `MAX_CONCURRENCY=2` 설정으로 상세 로그를 생성했고, 로그 파일도 `/data/arclang/logs/content_logging_solved.log`로 별도 저장했다. 메인 실험(PID 45157)은 중단 없이 계속 실행되었으며, content 로깅 실험(PID 3705300)은 task 695367ec를 성공적으로 완료했다(total count 1, correct count 1).

두 실험은 동일한 vLLM 서버(localhost:8000)를 공유했으며, vLLM의 PagedAttention 기반 자동 큐잉 시스템이 요청을 순차 처리하여 GPU CUDA OOM 없이 안정적으로 작동했다. content 로깅 결과, 모델은 각 문제 해결 단계마다 먼저 자연어로 규칙을 서술하는 "instructions" 필드를 생성한 후 이를 바탕으로 "grid" 필드에 실제 출력 배열을 생성하는 two-phase 전략을 사용함을 확인했다. 이는 GPT-OSS 20B가 implicit reasoning이 아닌 explicit rule articulation을 통해 문제를 해결함을 보여준다.
