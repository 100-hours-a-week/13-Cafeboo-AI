from fastapi import APIRouter
from pydantic import BaseModel
from ai_project.service.moderation_service import ModerationService

router = APIRouter()

class ModerationRequest(BaseModel):
    user_input: str

@router.post("/toxicity_detect")
async def moderate_input(request: ModerationRequest):
    service = ModerationService()
    result = service.moderate(request.user_input)
    return result