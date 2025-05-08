from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from models.coffee_recommendation import CoffeeRecommendationModel

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

# ✅ 응답 포맷 함수
def make_response(status: str, message: str, data: dict = None, code: int = 200):
    return {
        "status": status,
        "message": message,
        "data": data or {}
    }, code

@router.post("/coffee-recommendation/predict")
def recommend_coffee(request: CoffeeRecommendationRequest):
    try:
        # ✅ 필수값 누락 체크 (수동 예시)
        if request.caffeine_limit is None:
            return make_response(
                "error",
                "잘못된 요청입니다.",
                {
                    "code": "invalid_request",
                    "detail": "필수 파라미터 'caffeine_limit'이 누락되었습니다."
                },
                code=400
            )[0]

        model = CoffeeRecommendationModel()
        result = model.predict(request.dict())

        # ✅ 상태 변환
        caffeine_status = "Y" if result["can_drink"] else "N"

        return make_response(
            "success",
            "추가 커피 섭취 가능여부가 생성되었습니다.",
            {
                "user_id": request.user_id,
                "caffeine_status": caffeine_status
            },
            code=200
        )[0]

    except FileNotFoundError as e:
        return make_response(
            "error",
            "서버 내부 오류가 발생했습니다.",
            {
                "code": "resource_exhausted",
                "detail": str(e)
            },
            code=500
        )[0]

    except Exception as e:
        return make_response(
            "error",
            "서버 내부 오류가 발생했습니다.",
            {
                "code": "resource_exhausted",
                "detail": "컴퓨팅 리소스 부족으로 AI 추론을 완료할 수 없었습니다."
            },
            code=500
        )[0]
