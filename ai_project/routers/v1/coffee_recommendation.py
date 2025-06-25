from fastapi import APIRouter
from pydantic import BaseModel
from ai_project.models.coffee_recommendation import CoffeeRecommendationModel
from ai_project.exceptions import CustomHTTPException
import logging

# 로거 설정
logger = logging.getLogger(__name__)

router = APIRouter()

# ✅ 요청 바디
class CoffeeRecommendationRequest(BaseModel):
    user_id: str
    gender: str  # "M" or "F"
    age: int
    height: float
    weight: float
    is_smoker: int
    take_hormonal_contraceptive: int 
    caffeine_sensitivity: int
    current_caffeine: int
    caffeine_limit: int
    residual_at_sleep: float
    target_residual_at_sleep: float
    ##planned_caffeine_intake: int
    current_time: float
    sleep_time: float

# ✅ 응답 모델
class CoffeeRecommendationResponse(BaseModel):
    status: str
    message: str
    data: dict

##@router.post("/coffee-recommendation/predict")
@router.post("/internal/ai/can_intake_caffeine", response_model=CoffeeRecommendationResponse)
def recommend_coffee(request: CoffeeRecommendationRequest):
    try:
        logger.info(f"요청 처리 시작: user_id={request.user_id}")
        


        # 모델 예측
        model = CoffeeRecommendationModel()
        result = model.predict(request.dict())
        
        # 결과 반환
        caffeine_status = "Y" if result["can_drink"] else "N"
        logger.info(f"예측 완료: user_id={request.user_id}, caffeine_status={caffeine_status}")

        return CoffeeRecommendationResponse(
            status="success",
            message="추가 커피 섭취 가능여부가 생성되었습니다.",
            data={
                "user_id": request.user_id,
                "caffeine_status": caffeine_status
            }
        )

    except FileNotFoundError as e:
        logger.error(f"모델 파일 찾을 수 없음: {str(e)}")
        raise CustomHTTPException(
            status_code=500,
            code="resource_exhausted",
            message="서버 내부 오류가 발생했습니다.",
            detail=str(e)
        )

    except Exception as e:
        logger.error(f"예상치 못한 오류: {str(e)}", exc_info=True)
        raise CustomHTTPException(
            status_code=500,
            code="resource_exhausted",
            message="서버 내부 오류가 발생했습니다.",
            detail="컴퓨팅 리소스 부족으로 AI 추론을 완료할 수 없었습니다."
        )
