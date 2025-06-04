from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from ai_project.schemas.errors_schemas import ErrorResponse, ErrorData
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from ai_project.exceptions import CustomHTTPException
from ai_project.routers.v1.pdf_router import router as pdf_router 
from ai_project.routers.v1.caffeine_limit import router as caffeine_limit
##from ai_project.routers.v1.real_time_coffee import router as real_time_coffee
from ai_project.routers.v1.moderation_router import router as moderation_router
from ai_project.routers.v1.weekly_report_router import router as weekly_report_router
from ai_project.routers.v1.coffee_recommendation import router as coffee_recommendation

import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)

app = FastAPI()

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    error_response = ErrorResponse(
        status="error",
        message="잘못된 요청입니다.",
        data=ErrorData(
            code="invalid_request",
            detail=str(exc.errors()[0]["msg"])
        )
    )
    return JSONResponse(
        status_code=422,
        content=error_response.model_dump()
    )

# 글로벌 예외 핸들러 등록
@app.exception_handler(CustomHTTPException)
async def custom_exception_handler(request: Request, exc: CustomHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": exc.message,
            "data": exc.data
        }
    )

# HTTP 예외 핸들러
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": str(exc.detail),
            "data": {
                "code": f"HTTP_{exc.status_code}",
                "detail": {}
            }
        }
    )

# 일반 예외 핸들러
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "서버 내부 오류가 발생했습니다",
            "data": {
                "code": "INTERNAL_SERVER_ERROR",
                "detail": str(exc)
            }
        }
    )


app.include_router(caffeine_limit)
#app.include_router(real_time_coffee)
app.include_router(moderation_router, prefix="/internal/ai")
app.include_router(coffee_recommendation) 
app.include_router(pdf_router, prefix="/internal/ai")
app.include_router(weekly_report_router, prefix="/internal/ai")
