from fastapi import APIRouter

router = APIRouter()

@router.post("/weekly_report")
async def generate_report():
    """
    주간 리포트 생성
    """
    pass 