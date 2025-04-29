from fastapi import APIRouter

router = APIRouter()

@router.post("/real_time_coffee")
async def recommend_coffee():
    """
    실시간 커피 추천
    """
    pass 