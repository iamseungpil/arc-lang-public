# ARC-Lang Setup Guide

완벽하게 이식 가능한 설정 가이드입니다.

## 📋 목차
1. [기본 설정](#1-기본-설정)
2. [환경 변수 설정](#2-환경-변수-설정)
3. [실행 방법](#3-실행-방법)
4. [로컬 GPT-OSS 설정](#4-로컬-gpt-oss-설정)

---

## 1. 기본 설정

### Python 환경 준비

```bash
# Python 3.12 권장 (3.10 이상 필요)
python --version

# 의존성 설치
uv sync

# 또는 pip 사용
pip install -e .
```

### 환경 변수 파일 생성

```bash
# .env.example을 복사해서 사용
cp .env.example .env

# 편집기로 .env 파일 수정
nano .env  # 또는 vim, code 등
```

---

## 2. 환경 변수 설정

### 필수 설정

`.env` 파일에 다음을 **반드시** 설정:

```env
MAX_CONCURRENCY=4  # API 동시 호출 수 (필수!)
```

### 데이터 디렉토리 설정 (선택)

**옵션 A: 기본값 사용 (레포 내부)**
```env
# 아무것도 설정하지 않으면 ./attempts에 저장
```

**옵션 B: 커스텀 경로 사용**
```env
ARCLANG_DATA_DIR=/data/arclang  # 원하는 경로 지정
```

디렉토리는 자동으로 생성되므로 수동 생성 불필요!

### API 키 설정

사용할 모델 제공자의 API 키만 설정:

```env
# OpenAI GPT 사용 시
OPENAI_API_KEY=sk-...

# Anthropic Claude 사용 시
ANTHROPIC_API_KEY=sk-ant-...

# Google Gemini 사용 시
GEMINI_API_KEY=...

# xAI Grok 사용 시
XAI_API_KEY=key1,key2

# OpenRouter 사용 시 (GPT-OSS 120B 등)
OPENROUTER_API_KEY=sk-or-...
```

---

## 3. 실행 방법

### 방법 1: 직접 실행

```bash
# 환경 변수 설정 (.env 파일 사용)
export $(cat .env | grep -v '^#' | xargs)

# Python 실행
python src/run.py
```

### 방법 2: 스크립트 사용

```bash
# 실행 권한 부여 (최초 1회)
chmod +x run_experiment.sh

# 실행
./run_experiment.sh
```

### 실행 중 모니터링

```bash
# 진행 상황 확인
./monitor_progress.sh

# 중간 정확도 체크
python check_intermediate_accuracy.py
```

---

## 4. 로컬 GPT-OSS 설정

### 4.1. 사전 요구사항

- NVIDIA GPU 2개 (16GB+ VRAM 권장)
- CUDA 설치됨
- PyTorch 2.4+

### 4.2. vLLM 설치

```bash
# 옵션 A: uv 사용
uv sync --extra vllm

# 옵션 B: pip 사용
pip install 'vllm>=0.6.0,<0.7.0'
```

### 4.3. 환경 변수 설정

`.env`에 추가:

```env
# 로컬 vLLM 서버 URL
LOCAL_VLLM_URL=http://localhost:8000/v1

# GPU 설정 (선택사항, 기본값: 0,1)
VLLM_GPUS=0,1
VLLM_TENSOR_PARALLEL=2
VLLM_PORT=8000

# 동시성 낮추기 (로컬 서버는 리소스 제한적)
MAX_CONCURRENCY=4
```

### 4.4. vLLM 서버 시작

**터미널 1:**
```bash
chmod +x start_vllm_server.sh
./start_vllm_server.sh
```

서버가 준비되면 다음 메시지 출력:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 4.5. 실험 실행

**터미널 2:**
```bash
# src/run.py에서 config 확인
# 현재: local_gpt_oss_20b_config 사용 중

python src/run.py
```

### 4.6. GPU 모니터링

```bash
watch -n 1 nvidia-smi
```

---

## 🎯 빠른 시작 예제

### 예제 1: OpenRouter GPT-OSS 120B (클라우드)

```bash
# .env 설정
cat > .env <<EOF
MAX_CONCURRENCY=20
OPENROUTER_API_KEY=sk-or-your-key-here
EOF

# src/run.py 수정 (L1030)
# config=oss_config,  # OpenRouter GPT-OSS 120B

python src/run.py
```

### 예제 2: 로컬 GPT-OSS 20B

```bash
# .env 설정
cat > .env <<EOF
MAX_CONCURRENCY=4
LOCAL_VLLM_URL=http://localhost:8000/v1
EOF

# vLLM 서버 시작 (터미널 1)
./start_vllm_server.sh

# 실험 실행 (터미널 2)
# src/run.py에 이미 local_gpt_oss_20b_config 설정됨
python src/run.py
```

### 예제 3: Anthropic Claude

```bash
# .env 설정
cat > .env <<EOF
MAX_CONCURRENCY=120
ANTHROPIC_API_KEY=sk-ant-your-key-here
EOF

# src/run.py 수정 (L1030)
# config=sonnet_4_5_config_prod,

python src/run.py
```

---

## 📁 디렉토리 구조

실행 후 자동 생성되는 구조:

```
arc-lang-public/
├── attempts/                    # 기본 데이터 디렉토리
│   └── arc-prize-2024/
│       ├── arc-agi_evaluation_attempts.json
│       └── temp_solutions/
│           ├── 00d62c1b.json
│           └── ...
├── logs/                        # 로그 파일
│   └── vllm_server_*.log
└── ...

또는 (ARCLANG_DATA_DIR=/data/arclang 설정 시):

/data/arclang/
├── attempts/
│   └── arc-prize-2024/
└── logs/
```

---

## 🔧 트러블슈팅

### 문제: "MAX_CONCURRENCY not set"
```bash
# .env 파일에 추가
echo "MAX_CONCURRENCY=4" >> .env
```

### 문제: vLLM 서버 연결 실패
```bash
# 서버 실행 확인
curl http://localhost:8000/health

# 안 되면 서버 재시작
pkill -f vllm
./start_vllm_server.sh
```

### 문제: GPU 메모리 부족
```bash
# start_vllm_server.sh 수정
# --gpu-memory-utilization 0.85  →  0.75로 변경
```

### 문제: 디렉토리 권한 오류
```bash
# ARCLANG_DATA_DIR 사용 시 권한 확인
sudo chown -R $USER:$USER /data/arclang
```

---

## 📊 설정 비교표

| 설정 | OpenRouter GPT-OSS | 로컬 vLLM GPT-OSS | Claude Sonnet |
|------|-------------------|-------------------|---------------|
| MAX_CONCURRENCY | 40 | 4 | 120 |
| API 키 | OPENROUTER_API_KEY | 불필요 | ANTHROPIC_API_KEY |
| 하드웨어 | 없음 | GPU 2개 (32GB+) | 없음 |
| 비용 | 중간 | 무료 (전기세) | 높음 |
| 속도 | 빠름 | 느림 | 매우 빠름 |

---

## 📚 추가 문서

- **프레임워크 작동 방식**: `README.md`
- **프롬프트 엔지니어링**: `PROMPT_MECHANICS.md`
- **로컬 GPT-OSS 상세**: `LOCAL_GPT_OSS_SETUP.md`

---

## ✅ 설치 완료 체크리스트

- [ ] Python 3.10+ 설치됨
- [ ] 의존성 설치 완료 (`uv sync`)
- [ ] `.env` 파일 생성 및 MAX_CONCURRENCY 설정
- [ ] API 키 설정 (사용할 모델 제공자)
- [ ] (로컬 vLLM 사용 시) vLLM 설치 및 서버 시작
- [ ] `python src/run.py` 실행 성공

모든 항목이 체크되면 준비 완료! 🎉
