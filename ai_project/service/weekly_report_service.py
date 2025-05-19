import sys
from pathlib import Path


# 프로젝트 루트 경로를 찾고 sys.path에 추가
project_root = str(Path(__file__).resolve().parent.parent.parent)  # 13-Cafeboo-AI/ 디렉토리
sys.path.insert(0, project_root)
from langchain_huggingface import HuggingFaceEmbeddings
from ai_project.config.config import GOOGLE_API_KEY

from ai_project.pipelines.weekly_report_pipeline import WeeklyReportPipeline
from ai_project.schemas.weekly_report_schemas import BatchReportRequest, BatchReportCallbackData
from typing import Dict, Any, List
import logging
import asyncio
import httpx
from google import genai
import json
from langchain_chroma import Chroma
logger = logging.getLogger(__name__)
embedding_model_path = "ai_project/models/embedding_model"
class WeeklyReportService:
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_path,
            model_kwargs={"device": "cpu"}
        )
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        self.vectorstore = Chroma(
            collection_name="default_collection",
            embedding_function=self.embedding_model,
            persist_directory="chroma_db"
        )
        self.pipeline = WeeklyReportPipeline(embedding_model=self.embedding_model, client=self.client, vectorstore=self.vectorstore)
        
    async def generate_weekly_report(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        주간 카페인 소비 리포트를 생성합니다.
        
        Args:
            user_data: 사용자의 주간 카페인 소비 데이터
            
        Returns:
            Dict: 생성된 리포트와 상태 정보
        """
        try:
            # 파이프라인 실행은 CPU 바운드 작업이므로 별도 스레드에서 실행
           
            result = await self.pipeline.run({
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

    async def generate_weekly_reports(self, report_requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        여러 사용자의 주간 카페인 소비 리포트를 병렬로 생성합니다.
        
        Args:
            report_requests: 리포트 요청 데이터 목록
            
        Returns:
            List[Dict]: 각 사용자에 대한 리포트 결과 목록
        """
        logger.info(f"{len(report_requests)}명의 사용자에 대한 리포트 생성")

        # 모델 로드 및 주입 
        
        # 각 사용자에 대해 비동기 태스크 생성
        tasks = [self.generate_weekly_report(request) for request in report_requests]
        
        # 모든 태스크를 병렬로 실행하고 결과 수집
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 예외를 정상적인 에러 응답으로 변환
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"사용자 {report_requests[i]['user_id']}의 리포트 생성 중 예외 발생: {str(result)}")
                processed_results.append({
                    "status": "error",
                    "message": f"리포트 생성 중 예외가 발생했습니다: {str(result)}",
                    "report": None
                })
            else:
                processed_results.append(result)
                
        return processed_results

    async def generate_reports_and_callback(self, request: BatchReportRequest) -> None:
        try:
            logger.info(f"{len(request.users)}명의 사용자에 대한 리포트 생성 시작")
            
            # 사용자 데이터 변환
            report_requests = []
            for user in request.users:
                # 데이터 구조 변환 
                report_data = {
                    "user_id": user.user_id,
                    "period": user.data.get("period", ""),
                    "avg_caffeine_per_day": user.data.get("avg_caffeine_per_day", 0),
                    "recommended_daily_limit": user.data.get("recommended_daily_limit", 0),
                    "percentage_of_limit": user.data.get("percentage_of_limit", 0),
                    "highlight_day_high": user.data.get("highlight_day_high", ""),
                    "highlight_day_low": user.data.get("highlight_day_low", ""),
                    "first_coffee_avg": user.data.get("first_coffee_avg", ""),
                    "last_coffee_avg": user.data.get("last_coffee_avg", ""),
                    "late_night_caffeine_days": user.data.get("late_night_caffeine_days", 0),
                    "over_100mg_before_sleep_days": user.data.get("over_100mg_before_sleep_days", 0),
                    "average_sleep_quality": user.data.get("average_sleep_quality", "보통")  # 기본값 설정
                }
                report_requests.append(report_data)
            
            all_results = await self.generate_weekly_reports(report_requests)
            
            # 결과 포맷팅
            reports = []
            for i, result in enumerate(all_results):
                if result["status"] == "error":
                    logger.error(f"사용자 {report_requests[i]['user_id']}의 리포트 생성 중 오류: {result.get('message')}")
                    reports.append({
                        "user_id": report_requests[i]["user_id"],
                        "report": f"오류: {result.get('message', '리포트 생성 중 오류가 발생했습니다.')}"
                    })
                else:
                    reports.append({
                        "user_id": report_requests[i]["user_id"],
                        "report": result.get("report", "")
                    })
            
            # 콜백 URL로 결과 전송 
            async with httpx.AsyncClient() as client:
                callback_data = {"reports": reports}
                
                print("callback_data 구조:")
                print(json.dumps(callback_data, indent=2, ensure_ascii=False))

                # 상세 리포트 출력
                print("\n===== 생성된 리포트 상세 =====\n")
                for i, report_item in enumerate(callback_data["reports"], 1):
                    user_id = report_item.get("user_id", "ID 없음")
                    report_content = report_item.get("report", "")
                    
                    print(f"[리포트 #{i} - 사용자: {user_id}]")
                    print("-" * 50)
                    print(report_content)
                    print("\n" + "=" * 70 + "\n")
                
                # 콜백은 한 번만 수행
                logger.info(f"{request.callback_url}로 {len(reports)}개의 리포트 결과 전송")
                try:
                    response = await client.post(str(request.callback_url), json=callback_data, timeout=30.0)
                    logger.info(f"콜백 응답 상태 코드: {response.status_code}")
                except Exception as e:
                    logger.error(f"콜백 전송 실패: {str(e)}")
                    # 콜백 실패해도 계속 진행
            
        except Exception as e:
            logger.error(f"리포트 생성 및 콜백 처리 중 오류: {str(e)}", exc_info=True)
            # 오류 발생 시에도 콜백 시도
            try:
                async with httpx.AsyncClient() as client:
                    error_data = {"reports": [], "error": str(e)}
                    await client.post(str(request.callback_url), json=error_data, timeout=30.0)
            except Exception as callback_error:
                logger.error(f"오류 콜백 전송 실패: {str(callback_error)}")
if __name__ == "__main__":
    import asyncio
    from typing import List, Dict, Any
    
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    async def test_batch_reports():
        print("배치 리포트 생성 테스트 시작")
        
        # 테스트용 사용자 데이터 생성
        test_users = [
            {
                "user_id": "test_user1",
                "period": "2025-04-01 ~ 04-07",
                "avg_caffeine_per_day": 170,
                "recommended_daily_limit": 300,
                "percentage_of_limit": 57,
                "highlight_day_high": "수요일",
                "highlight_day_low": "금요일",
                "first_coffee_avg": "09:20",
                "last_coffee_avg": "16:45",
                "late_night_caffeine_days": 2,
                "over_100mg_before_sleep_days": 1,
                "average_sleep_quality": "좋음"
            },
            {
                "user_id": "test_user2",
                "period": "2025-04-01 ~ 04-07",
                "avg_caffeine_per_day": 250,
                "recommended_daily_limit": 300,
                "percentage_of_limit": 83,
                "highlight_day_high": "월요일",
                "highlight_day_low": "일요일",
                "first_coffee_avg": "08:00",
                "last_coffee_avg": "17:30",
                "late_night_caffeine_days": 3,
                "over_100mg_before_sleep_days": 2,
                "average_sleep_quality": "보통"
            },
            {
                "user_id": "test_user3",
                "period": "2025-04-01 ~ 04-07",
                "avg_caffeine_per_day": 100,
                "recommended_daily_limit": 300,
                "percentage_of_limit": 33,
                "highlight_day_high": "화요일",
                "highlight_day_low": "목요일",
                "first_coffee_avg": "10:30",
                "last_coffee_avg": "15:00",
                "late_night_caffeine_days": 0,
                "over_100mg_before_sleep_days": 0,
                "average_sleep_quality": "매우 좋음"
            },
            {
                "user_id": "test_user4",
                "period": "2025-04-01 ~ 04-07",
                "avg_caffeine_per_day": 100,
                "recommended_daily_limit": 300,
                "percentage_of_limit": 33,
                "highlight_day_high": "화요일",
                "highlight_day_low": "목요일",
                "first_coffee_avg": "10:30",
                "last_coffee_avg": "15:00",
                "late_night_caffeine_days": 0,
                "over_100mg_before_sleep_days": 0,
                "average_sleep_quality": "매우 좋음"
            },
            {
                "user_id": "test_user5",
                "period": "2025-04-01 ~ 04-07",
                "avg_caffeine_per_day": 100,
                "recommended_daily_limit": 300,
                "percentage_of_limit": 33,
                "highlight_day_high": "화요일",
                "highlight_day_low": "목요일",
                "first_coffee_avg": "10:30",
                "last_coffee_avg": "15:00",
                "late_night_caffeine_days": 0,
                "over_100mg_before_sleep_days": 0,
                "average_sleep_quality": "매우 좋음"
            },
            {
                "user_id": "test_user6",
                "period": "2025-04-01 ~ 04-07",
                "avg_caffeine_per_day": 100,
                "recommended_daily_limit": 300,
                "percentage_of_limit": 33,
                "highlight_day_high": "화요일",
                "highlight_day_low": "목요일",
                "first_coffee_avg": "10:30",
                "last_coffee_avg": "15:00",
                "late_night_caffeine_days": 0,
                "over_100mg_before_sleep_days": 0,
                "average_sleep_quality": "매우 좋음"
            },
            {
                "user_id": "test_user7",
                "period": "2025-04-01 ~ 04-07",
                "avg_caffeine_per_day": 100,
                "recommended_daily_limit": 300,
                "percentage_of_limit": 33,
                "highlight_day_high": "화요일",
                "highlight_day_low": "목요일",
                "first_coffee_avg": "10:30",
                "last_coffee_avg": "15:00",
                "late_night_caffeine_days": 0,
                "over_100mg_before_sleep_days": 0,
                "average_sleep_quality": "매우 좋음"
            },{
                "user_id": "test_user8",
                "period": "2025-04-01 ~ 04-07",
                "avg_caffeine_per_day": 100,
                "recommended_daily_limit": 300,
                "percentage_of_limit": 33,
                "highlight_day_high": "화요일",
                "highlight_day_low": "목요일",
                "first_coffee_avg": "10:30",
                "last_coffee_avg": "15:00",
                "late_night_caffeine_days": 0,
                "over_100mg_before_sleep_days": 0,
                "average_sleep_quality": "매우 좋음"
            },{
                "user_id": "test_user9",
                "period": "2025-04-01 ~ 04-07",
                "avg_caffeine_per_day": 100,
                "recommended_daily_limit": 300,
                "percentage_of_limit": 33,
                "highlight_day_high": "화요일",
                "highlight_day_low": "목요일",
                "first_coffee_avg": "10:30",
                "last_coffee_avg": "15:00",
                "late_night_caffeine_days": 0,
                "over_100mg_before_sleep_days": 0,
                "average_sleep_quality": "매우 좋음"
            },
            {
                "user_id": "test_user10",
                "period": "2025-04-01 ~ 04-07",
                "avg_caffeine_per_day": 100,
                "recommended_daily_limit": 300,
                "percentage_of_limit": 33,
                "highlight_day_high": "화요일",
                "highlight_day_low": "목요일",
                "first_coffee_avg": "10:30",
                "last_coffee_avg": "15:00",
                "late_night_caffeine_days": 0,
                "over_100mg_before_sleep_days": 0,
                "average_sleep_quality": "매우 좋음"
            },{
                "user_id": "test_user11",
                "period": "2025-04-01 ~ 04-07",
                "avg_caffeine_per_day": 100,
                "recommended_daily_limit": 300,
                "percentage_of_limit": 33,
                "highlight_day_high": "화요일",
                "highlight_day_low": "목요일",
                "first_coffee_avg": "10:30",
                "last_coffee_avg": "15:00",
                "late_night_caffeine_days": 0,
                "over_100mg_before_sleep_days": 0,
                "average_sleep_quality": "매우 좋음"
            },
            {
                "user_id": "test_user12",
                "period": "2025-04-01 ~ 04-07",
                "avg_caffeine_per_day": 100,
                "recommended_daily_limit": 300,
                "percentage_of_limit": 33,
                "highlight_day_high": "화요일",
                "highlight_day_low": "목요일",
                "first_coffee_avg": "10:30",
                "last_coffee_avg": "15:00",
                "late_night_caffeine_days": 0,
                "over_100mg_before_sleep_days": 0,
                "average_sleep_quality": "매우 좋음"
            },
            {
                "user_id": "test_user13",
                "period": "2025-04-01 ~ 04-07",
                "avg_caffeine_per_day": 100,
                "recommended_daily_limit": 300,
                "percentage_of_limit": 33,
                "highlight_day_high": "화요일",
                "highlight_day_low": "목요일",
                "first_coffee_avg": "10:30",
                "last_coffee_avg": "15:00",
                "late_night_caffeine_days": 0,
                "over_100mg_before_sleep_days": 0,
                "average_sleep_quality": "매우 좋음"
            },
            {
                "user_id": "test_user14",
                "period": "2025-04-01 ~ 04-07",
                "avg_caffeine_per_day": 100,
                "recommended_daily_limit": 300,
                "percentage_of_limit": 33,
                "highlight_day_high": "화요일",
                "highlight_day_low": "목요일",
                "first_coffee_avg": "10:30",
                "last_coffee_avg": "15:00",
                "late_night_caffeine_days": 0,
                "over_100mg_before_sleep_days": 0,
                "average_sleep_quality": "매우 좋음"
            },
            {
                "user_id": "test_user15",
                "period": "2025-04-01 ~ 04-07",
                "avg_caffeine_per_day": 100,
                "recommended_daily_limit": 300,
                "percentage_of_limit": 33,
                "highlight_day_high": "화요일",
                "highlight_day_low": "목요일",
                "first_coffee_avg": "10:30",
                "last_coffee_avg": "15:00",
                "late_night_caffeine_days": 0,
                "over_100mg_before_sleep_days": 0,
                "average_sleep_quality": "매우 좋음"
            },
            {
                "user_id": "test_user16",
                "period": "2025-04-01 ~ 04-07",
                "avg_caffeine_per_day": 100,
                "recommended_daily_limit": 300,
                "percentage_of_limit": 33,
                "highlight_day_high": "화요일",
                "highlight_day_low": "목요일",
                "first_coffee_avg": "10:30",
                "last_coffee_avg": "15:00",
                "late_night_caffeine_days": 0,
                "over_100mg_before_sleep_days": 0,
                "average_sleep_quality": "매우 좋음"
            }
        ]
        
        print(f"{len(test_users)}명의 사용자에 대한 리포트 생성 테스트")
        
        # 서비스 인스턴스 생성
        service = WeeklyReportService()
        
        # 배치 리포트 생성
        results = await service.generate_weekly_reports(test_users)
        
        # 결과 출력
        print("\n===== 생성된 리포트 =====")
        for i, result in enumerate(results):
            user_id = test_users[i]["user_id"]
            status = result.get("status", "unknown")
            
            print(f"\n[사용자 {user_id}] 상태: {status}")
            
            if status == "success":
                report = result.get("report", "")
                preview = report[:150] + "..." if len(report) > 150 else report
                print(f"리포트 미리보기: {preview}")
            else:
                error = result.get("message", "알 수 없는 오류")
                print(f"오류: {error}")
        
    # 테스트 실행
    asyncio.run(test_batch_reports())
      
            
