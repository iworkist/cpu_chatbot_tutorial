# CPU Chatbot Tutorial

LLM API를 사용한 챗봇 예제 모음 (OpenAI, Ollama, Cerebras)

## 설치

```bash
uv sync
```

## 환경 변수 설정

`.env` 파일 생성 또는 환경 변수 설정:

```bash
export OPENAI_API_KEY="your_api_key_here"
export CEREBRAS_API_KEY="your_api_key_here"  # Cerebras 사용 시
```

## 실행

```bash
# OpenAI 스트리밍 챗봇
uv run python 01_LLM_API/eg_openai_stream_chatbot.py

# Ollama 스트리밍 챗봇 (로컬)
uv run python 01_LLM_API/eg_ollama_stream_chatbot.py

# Cerebras 스트리밍 챗봇
uv run python 01_LLM_API/eg_cerebras_stream_chatbot.py

# OpenAI 모델 목록 조회
uv run python 01_LLM_API/eg_openai_models.py
```
