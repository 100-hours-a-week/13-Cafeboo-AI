from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from ai_project.schemas.drink_schema import UserFeatures
from ai_project.service.drink_recommendation_loader import drink_recommender
from ai_project.utils.common import make_response
from ai_project.exceptions import CustomHTTPException

router = APIRouter(prefix="/internal/ai")

@router.post("/drink_recommendation")
async def recommend_drinks(user: UserFeatures):
    try:
        recs = drink_recommender.recommend(user.dict(), top_n=3)
        return make_response(
            status="success",
            message="상위 3개의 drink_id가 반환되었습니다.",
            data={
                "status": "success",
                "data": {"drink_ids": recs}
            }
        )[0]
    
    except KeyError as ke:
        return CustomHTTPException(
            status_code=400,
            code="invalid_request",
            message="잘못된 요청입니다.",
            detail=f"필수 파라미터 '{ke.args[0]}'가 누락되었거나 잘못되었습니다."
        )            

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