from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any, Optional
from ai_project.service.weekly_report_service import WeeklyReportService
import logging

logger = logging.getLogger(__name__)

# 요청 모델 정의
class CaffeineWeeklyReportRequest(BaseModel):
    user_id: str
    period: str
    avg_caffeine_per_day: float
    recommended_daily_limit: float
    percentage_of_limit: float
    highlight_day_high: str
    highlight_day_low: str
    first_coffee_avg: str
    last_coffee_avg: str
    late_night_caffeine_days: int
    over_100mg_before_sleep_days: int
    average_sleep_quality: str

# 응답 데이터 모델
class ReportData(BaseModel):
    user_id: str
    report: str
    

# 응답 모델 정의
class CaffeineWeeklyReportResponse(BaseModel):
    status: str
    message: str
    data: Optional[ReportData] = None

router = APIRouter()

def get_weekly_report_service():
    return WeeklyReportService()

@router.post("/caffeine_weekly_report", response_model=CaffeineWeeklyReportResponse)
async def generate_caffeine_weekly_report(
    request: CaffeineWeeklyReportRequest,
    report_service: WeeklyReportService = Depends(get_weekly_report_service)
):
    """
    사용자의 커피 소비 데이터를 기반으로 주간 카페인 소비 리포트를 생성합니다.
    """
    # 요청 데이터 로깅
    logger.info(f"Received weekly report request for user_id: {request.user_id}, period: {request.period}")
    
    # 요청 모델을 dict로 변환
    user_data = request.dict()
    
    # 서비스 호출
    result = report_service.generate_weekly_report(user_data)
    
    # 에러 처리는 상위 레벨에서 처리
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result.get("message", "리포트 생성 중 오류가 발생했습니다."))
    
    # 성공 응답
    return CaffeineWeeklyReportResponse(
        status="success",
        message="리포트 생성을 완료했습니다.",
        data=ReportData(
            user_id=request.user_id,
            report=result.get("report", "")
        )
    ) 