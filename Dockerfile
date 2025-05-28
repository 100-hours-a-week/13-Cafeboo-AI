# Python 3.12.7 기반 이미지
FROM python:3.12.7-slim

# 작업 디렉토리 생성
WORKDIR /app

# 🔧 필수 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# requirements.txt 복사 및 설치
COPY ai_project/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 전체 코드 복사
COPY . .

# FastAPI 실행 (내부 통신용)
CMD ["uvicorn", "ai_project.main:app", "--host", "0.0.0.0", "--port", "8000"]
