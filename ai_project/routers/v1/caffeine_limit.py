from fastapi import APIRouter

router = APIRouter()

@router.post("/caffeine_limit")
async def predict_limit():
    """
    하루 카페인 한도 예측
    """
    pass 