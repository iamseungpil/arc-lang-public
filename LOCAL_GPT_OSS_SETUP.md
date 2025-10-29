# Local GPT-OSS 20B Setup Guide

이 가이드는 로컬 GPU 0, 1번에서 GPT-OSS 20B를 실행하고 ARC 실험을 진행하는 방법을 설명합니다.

## 📋 구현 완료 사항

### 1. 코드 수정
- ✅ `src/llms/models.py`: `local_gpt_oss_20b` 모델 추가
- ✅ `src/llms/structured.py`: 로컬 vLLM 서버 호출 함수 추가
- ✅ `src/configs/oss_configs.py`: `local_gpt_oss_20b_config` 설정 추가
- ✅ `src/run.py`: `/data/arclang` 경로 설정 및 `local_gpt_oss_20b_config` 사용

### 2. 디렉토리 구조
```
/data/arclang/
├── logs/              # vLLM 서버 로그
├── checkpoints/       # 체크포인트
└── attempts/          # ARC 실험 결과
    └── arc-prize-2024/
```

## 🚀 사용 방법

### Step 1: vLLM 서버 시작

**터미널 1에서 실행:**
```bash
cd /home/ubuntu/arc-lang-public
./start_vllm_server.sh
```

이 스크립트는:
- vLLM을 자동 설치 (필요시)
- GPU 0, 1에서 tensor parallel로 GPT-OSS 20B 서버 시작
- 포트 8000에서 OpenAI API 호환 서버 실행
- 로그를 `/data/arclang/logs/` 에 저장

**서버가 준비되면 다음과 같은 메시지가 표시됩니다:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 2: ARC 실험 실행

**터미널 2에서 실행:**
```bash
cd /home/ubuntu/arc-lang-public
export MAX_CONCURRENCY=20
export LOCAL_VLLM_URL="http://localhost:8000/v1"  # 기본값이므로 생략 가능
python src/run.py
```

## 🔧 환경 변수

필요한 환경 변수를 `.env` 파일에 추가하세요:

```bash
# 기존 환경 변수들...
MAX_CONCURRENCY=20

# 로컬 vLLM 서버 URL (선택사항, 기본값: http://localhost:8000/v1)
LOCAL_VLLM_URL=http://localhost:8000/v1
```

## 📊 현재 설정

`src/run.py` 의 현재 설정:
- **Config**: `local_gpt_oss_20b_config`
- **Dataset**: ARC 2024 evaluation
- **Limit**: 1 task (테스트용)
- **Offset**: 0

실제 실험을 위해 `src/run.py`의 `limit` 값을 조정하세요:
```python
await run_from_json(
    challenges_path=challenges_path,
    truth_solutions_path=solutions_path,
    config=local_gpt_oss_20b_config,
    attempts_path=attempts_path,
    temp_attempts_dir=temp_attempts_path,
    limit=None,  # 전체 데이터셋
    offset=0,
)
```

## 🔍 모니터링

### vLLM 서버 로그 확인
```bash
tail -f /data/arclang/logs/vllm_server_*.log
```

### GPU 사용률 확인
```bash
watch -n 1 nvidia-smi
```

### 실험 결과 확인
```bash
ls -lh /data/arclang/attempts/arc-prize-2024/
```

## 🛠️ 트러블슈팅

### vLLM 서버가 시작되지 않을 때
1. GPU 메모리 확인: `nvidia-smi`
2. 포트 사용 확인: `netstat -tlnp | grep 8000`
3. 로그 확인: `tail -f /data/arclang/logs/vllm_server_*.log`

### 연결 오류가 발생할 때
1. vLLM 서버가 실행 중인지 확인
2. `LOCAL_VLLM_URL` 환경 변수 확인
3. 방화벽 설정 확인

### Out of Memory 오류
`start_vllm_server.sh`에서 `--gpu-memory-utilization` 값을 낮추세요:
```bash
--gpu-memory-utilization 0.8  # 기본값 0.95에서 0.8로 변경
```

## 📝 기술 세부사항

### vLLM vs Transformers 직접 사용
vLLM을 선택한 이유:
- **24배 빠른 처리량** (2025 벤치마크)
- **배치 처리 최적화**: PagedAttention으로 메모리 효율적
- **OpenAI API 호환**: 최소한의 코드 변경
- **Tensor Parallel 지원**: GPU 0, 1 동시 활용

### 모델 정보
- **Model**: `openai/gpt-oss-20b`
- **Parameters**: 21B (MoE, 3.6B active)
- **Quantization**: MXFP4 (4-bit)
- **메모리**: ~16GB (GPU당 충분히 여유)

## 🔄 기존 OpenRouter 설정과의 비교

기존 OpenRouter GPT-OSS 120B 설정은 그대로 유지됩니다:
- `oss_config`: OpenRouter를 통한 GPT-OSS 120B
- `local_gpt_oss_20b_config`: 로컬 vLLM을 통한 GPT-OSS 20B

필요에 따라 `src/run.py`에서 config를 변경하여 사용하세요.
