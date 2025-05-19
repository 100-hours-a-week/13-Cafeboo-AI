from pydantic import BaseModel, HttpUrl
from typing import List, Dict, Any, Optional

# 개별 리포트 요청 모델
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

# 배치 처리를 위한 모델
class UserReportData(BaseModel):
    user_id: str
    data: Dict[str, Any]

class BatchReportRequest(BaseModel):
    callback_url: HttpUrl
    users: List[UserReportData]

# 응답 데이터 모델
class ReportData(BaseModel):
    user_id: str
    report: str

# 개별 리포트 응답 모델
class CaffeineWeeklyReportResponse(BaseModel):
    status: str
    message: str
    data: Optional[ReportData] = None

# 배치 리포트 응답 모델
class BatchReportResponse(BaseModel):
    status: str
    message: str
    #task_id: str

# 콜백 데이터 모델
class BatchReportCallbackData(BaseModel):
    reports: List[Dict[str, str]]