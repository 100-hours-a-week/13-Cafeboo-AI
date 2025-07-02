from fastapi import APIRouter
from pydantic import BaseModel
from ai_project.exceptions import CustomHTTPException
from ai_project.service.moderation_service import moderation_service
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

class ModerationRequest(BaseModel):
    user_input: str

class ModerationResponse(BaseModel):
    status: str
    message: str
    is_toxic: int



@router.post("/toxicity_detect", response_model=ModerationResponse)
async def moderate_input(request: ModerationRequest):
    
    try:
        result = moderation_service.moderate(request.user_input)
        return ModerationResponse(status="success", message=result["message"], is_toxic=result["is_toxic"])
    except Exception as e:
        logger.error(f"유해성 검사 중 오류 발생: {str(e)}")
        raise CustomHTTPException(
            status_code=500,
            code="MODERATION_ERROR",
            message="유해성 검사 중 오류가 발생했습니다",
            detail= str(e)
        )