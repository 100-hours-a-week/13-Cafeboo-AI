from fastapi import APIRouter

router = APIRouter()

@router.post("/coffee_recommendation")
async def recommend_coffee():
    """
    커피 추천
    """
    pass 