from ai_project.pipelines.pdf_pipeline import PDFProcessingPipeline, create_pdf_processing_workflow
from typing import Dict

class PDFService:
    def __init__(self):
        self.pipeline = PDFProcessingPipeline()
        self.workflow = create_pdf_processing_workflow()
        
    def process_pdf(self, pdf_path: str)-> Dict:
        return self.workflow.invoke({
            "pdf_path": pdf_path,
        })
    
    