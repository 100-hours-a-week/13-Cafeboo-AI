from fastapi import APIRouter

router = APIRouter()

@router.post("/input_moderation")
async def moderate_input():
    """
    입력 검열
    """
    pass 