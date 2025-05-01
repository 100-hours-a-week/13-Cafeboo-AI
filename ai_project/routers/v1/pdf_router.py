from fastapi import APIRouter
from ai_project.service.pdf_service import PDFService
from fastapi import Depends

router = APIRouter(prefix="/pdf", tags=["PDF"])

def get_pdf_service():
    return PDFService()

@router.post("/upload_pdf")
async def upload_pdf(file_path: str, pdf_service = Depends(get_pdf_service)):  
    return pdf_service.process_pdf(file_path)

