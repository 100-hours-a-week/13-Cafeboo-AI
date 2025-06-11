# 1. Base: OS 의존성 + core Python 패키지
FROM python:3.12.7-slim AS base
WORKDIR /app
RUN apt-get update \
  && apt-get install -y curl tar \
  && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir google-genai

# 2. Dependencies: requirements 설치
FROM base AS deps
COPY ai_project/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Models: embedding & moderation 모델 다운로드
FROM deps AS models

COPY ai_project/models /app/ai_project/models

RUN curl -L -o embedding_model.tar.gz https://storage.googleapis.com/ai_model_cafeboo/embedding_model.tar.gz \
 && tar -xzf embedding_model.tar.gz -C /app/ai_project/models \
 && rm embedding_model.tar.gz
 
RUN curl -L -o /app/ai_project/models/best_model.pt https://storage.googleapis.com/ai_model_cafeboo/moderation_model/best_model.pt

# 4. Final: 애플리케이션 코드 복사 및 실행 커맨드
FROM models AS final
WORKDIR /app
COPY . .
CMD ["uvicorn", "ai_project.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "debug"]