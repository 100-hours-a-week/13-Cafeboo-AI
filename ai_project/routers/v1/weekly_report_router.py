from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, Any
from ai_project.service.weekly_report_service import WeeklyReportService
from ai_project.schemas.weekly_report_schemas import (
    CaffeineWeeklyReportRequest,
    CaffeineWeeklyReportResponse,
    BatchReportRequest,
    BatchReportResponse,
    ReportData
)
import logging
import uuid

logger = logging.getLogger(__name__)

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
    result = await report_service.generate_weekly_report(user_data)
    
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

@router.post("/caffeine_weekly_reports", response_model=BatchReportResponse, status_code=202)
async def generate_caffeine_weekly_reports(
    request: BatchReportRequest,
    background_tasks: BackgroundTasks,
    report_service: WeeklyReportService = Depends(get_weekly_report_service),
):
    """
    사용자의 커피 소비 데이터를 기반으로 주간 카페인 소비 리포트들을 생성합니다.
    """
    try:
        # 유효성 검사
        # 서비스단에 유효성 검사 메소드 만들어서 추가
        # 백그라운드에서 리포트 생성 및 콜백 처리
        background_tasks.add_task(report_service.generate_reports_and_callback, request)

        # 응답 먼저 반환
        return BatchReportResponse(
            status="success",
            message="리포트 생성 요청을 받았습니다.",
            #task_id=str(uuid.uuid4())
        )

    except HTTPException as e:
        # 서비스단에서 발생한 400/500 에러 그대로 전파
        raise e