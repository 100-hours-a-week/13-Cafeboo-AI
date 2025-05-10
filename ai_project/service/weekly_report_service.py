from ai_project.pipelines.weekly_report_pipeline import WeeklyReportPipeline
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class WeeklyReportService:
    def __init__(self):
        self.pipeline = WeeklyReportPipeline()
        
    def generate_weekly_report(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        주간 카페인 소비 리포트를 생성합니다.
        
        Args:
            user_data: 사용자의 주간 카페인 소비 데이터
            
        Returns:
            Dict: 생성된 리포트와 상태 정보
        """
        try:
            # 파이프라인에 입력 데이터 전달
            result = self.pipeline.run({
                "user_input": user_data,
                "collection_name": "default_collection"
            })
            
            logger.info(f"Weekly report pipeline result status: {result.get('status', 'unknown')}")
            
            # 결과 처리
            if result.get("status", "").startswith("error"):
                logger.error(f"Weekly report generation error: {result.get('error', 'Unknown error')}")
                return {
                    "status": "error",
                    "message": result.get("error", "리포트 생성 중 오류가 발생했습니다."),
                    "report": None
                }
            
            # 성공적으로 리포트 생성된 경우
            return {
                "status": "success",
                "report": result.get("final_report", ""),
                "groundedness": result.get("groundedness_result", {}).get("status", "unknown")
            }
            
        except Exception as e:
            logger.error(f"Error in generate_weekly_report: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"리포트 생성 중 예외가 발생했습니다: {str(e)}",
                "report": None
            } 