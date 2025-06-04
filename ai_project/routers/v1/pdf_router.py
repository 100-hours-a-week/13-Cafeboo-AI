from fastapi import APIRouter, HTTPException, Request
from ai_project.service.pdf_service import PDFService
from fastapi import Depends
from pydantic import BaseModel
from typing import Dict, Optional
import logging
from fastapi.responses import JSONResponse
from ai_project.exceptions import CustomHTTPException

logger = logging.getLogger(__name__)

# 요청 모델
class PDFRequest(BaseModel):
    file_path: str

# 응답 모델
class PDFResponse(BaseModel):
    status: str
    collection_name: str
    document_count: int
    message: Optional[str] = None

router = APIRouter(prefix="/pdf", tags=["PDF"])

def get_pdf_service():
    return PDFService()

@router.post("/upload_pdf", response_model=PDFResponse)
async def upload_pdf(
    request: PDFRequest,
    pdf_service: PDFService = Depends(get_pdf_service)
):
    try:
        
        result = pdf_service.process_pdf(request.file_path)
        
        
        response = PDFResponse(
            status="success",
            collection_name=result.get("collection_name", "default_collection"),
            document_count=result.get("document_count", 0),
            message="PDF 처리 완료"
        )
        return response

    except FileNotFoundError:
        logger.error(f"파일을 찾을 수 없습니다: {request.file_path}")
        raise CustomHTTPException(
            status_code=404,
            code="PDF_NOT_FOUND",
            message=f"PDF 파일을 찾을 수 없습니다: {request.file_path}",
            detail={"file_path": request.file_path}
        )
    except Exception as e:
        logger.error(f"PDF 처리 중 오류 발생: {str(e)}")
        raise CustomHTTPException(
            status_code=500,
            code="PDF_PROCESSING_ERROR",
            message=f"PDF 처리 중 오류가 발생했습니다",
            detail={"error": str(e)}
        )