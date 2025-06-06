# Python 3.12.7 기반 이미지
FROM python:3.12.7-slim

# 작업 디렉토리 생성
WORKDIR /app

# 필수 시스템 유틸 설치
RUN apt-get update && apt-get install -y curl tar && rm -rf /var/lib/apt/lists/*

# 구글 관련 패키지 설치
RUN pip install google-genai

# requirements.txt 복사 및 설치
COPY ai_project/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 전체 코드 복사
COPY . .

# embedding_model.tar 다운로드 및 압축 해제
RUN curl -L -o embedding_model.tar https://storage.googleapis.com/ai_model_cafeboo/embedding_model.tar && \
    tar -xzf embedding_model.tar -C /app/ai_project/models && \
    rm embedding_model.tar

# moderation_model 추가
RUN curl -L -o /app/ai_project/models/best_model.pt https://storage.googleapis.com/ai_model_cafeboo/moderation_model/best_model.pt



# FastAPI 실행 (내부 통신용)
CMD ["uvicorn", "ai_project.main:app", "--host", "0.0.0.0", "--port", "8000"]
