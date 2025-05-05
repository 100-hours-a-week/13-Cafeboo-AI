from ai_project.pipelines.pdf_pipeline import PDFProcessingPipeline, create_pdf_processing_workflow
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class PDFService:
    def __init__(self):
        self.pipeline = PDFProcessingPipeline()
        self.workflow = create_pdf_processing_workflow()
        
    def process_pdf(self, pdf_path: str) -> Dict:
        try:
            result = self.workflow.invoke({
                "pdf_path": pdf_path,
            })
            
            
            logger.info(f"Workflow result: {result}")

            
            # 결과가 None이거나 필요한 키가 없는 경우 기본값 설정
            if result is None:
                result = {}
            
            # 확실하게 필수 필드를 포함하여 반환
            return {
                "status": "success",
                "collection_name": result.get("collection_name", "default_collection"),
                "document_count": result.get("document_count", 0)
            }
        except Exception as e:
            logger.error(f"Error in process_pdf: {e}", exc_info=True)
            raise
    
    