from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ai_project.models.caffeine_limit import CaffeineLimitModel
from ai_project.utils.common import make_response

router = APIRouter()

class CaffeineLimitRequest(BaseModel):
    user_id: str
    gender: str  # "M" or "F"
    age: int
    height: float
    weight: float
    is_smoker: int
    is_pregnant: int
    take_hormonal_contraceptive: int
    caffeine_sensitivity: int
    ##total_caffeine_today: int
    ##caffeine_intake_count: int
    ##first_intake_hour: int
    ##last_intake_hour: int
    ##sleep_duration: float
    ##sleep_quality: str  # "bad", "normal", "good"

##@router.post("/caffeine-limit/predict")
@router.post("/internal/ai/predict_limit")
def predict_caffeine_limit(request: CaffeineLimitRequest):
    try:
        model = CaffeineLimitModel()

        caffeine_limit = model.predict(request.dict())

        return make_response(
            status="success",
            message="유저에게 맞는 최대 카페인량이 계산되었습니다.",
            data={
                "user_id": request.user_id,
                "max_caffeine_mg": round(caffeine_limit)
            }
        )[0]
    except FileNotFoundError as e:
        raise CustomHTTPException(
            status_code=500,
            code="internal_server_error",
            message="서버 내부 오류가 발생했습니다.",
            detail=str(e)
        )
    except Exception as e:
        raise CustomHTTPException(
            status_code=500,
            code="unexpected_error",
            message="서버 내부 오류가 발생했습니다.",
            detail=str(e)
        )
