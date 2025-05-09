import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 환경 변수에서 불러오기
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# 모델 경로
MODEL_PATH = os.getenv("MODEL_PATH")

# 기타 설정
DEBUG = os.getenv("DEBUG", "False").lower() == "true" 