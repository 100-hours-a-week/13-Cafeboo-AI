# 1. Base: OS + Python 환경
FROM python:3.12.7-slim AS base
WORKDIR /app
RUN apt-get update \
  && apt-get install -y curl tar \
  && rm -rf /var/lib/apt/lists/*

# 2. Deps: pip requirements 설치
FROM base AS deps
COPY ai_project/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Models: 모델 파일 다운로드만 수행
FROM deps AS models
WORKDIR /app/ai_project/models
RUN curl -L -o embedding_model.tar.gz https://storage.googleapis.com/ai_model_cafeboo/embedding_model.tar.gz \
 && tar -xzf embedding_model.tar.gz \
 && rm embedding_model.tar.gz \
 && curl -L -o best_model.pt https://storage.googleapis.com/ai_model_cafeboo/moderation_model/best_model.pt

# 4. Final: 최종 이미지 - 실행에 필요한 것만 복사
FROM models AS final

WORKDIR /app

# (1) 의존성 복사
COPY --from=deps /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=deps /usr/local/bin /usr/local/bin

# (2) 모델 복사
COPY --from=models /app/ai_project/models /app/ai_project/models

# (3) 코드 복사
COPY . .

# 실행
CMD ["uvicorn", "ai_project.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "debug"]