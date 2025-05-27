from typing import Dict
import torch
from langgraph.graph import Graph
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import logging
from google import genai
from llama_cloud_services import LlamaParse
from langchain.schema import Document
import json
from ai_project.config.config import LLAMA_API_KEY, GOOGLE_API_KEY

logger = logging.getLogger(__name__)

class PDFProcessingPipeline:
    def __init__(self):
        self.client = genai.Client(api_key=GOOGLE_API_KEY)

        self.embeddings = HuggingFaceEmbeddings(
            model_name="ai_project/models/embedding_model",  
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True}  # 임베딩 정규화 추가
        )
        # self.text_splitter = RecursiveCharacterTextSplitter(
        #     chunk_size=500,
        #     chunk_overlap=50
        # )
        self.text_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("##", "section"), ("###", "subsection")]
        )
        self.parser = LlamaParse(
            api_key=LLAMA_API_KEY,         
            preset="complexTables",
            verbose="True" 
            )
        
    def load_pdf(self, state: Dict) -> Dict:
        try:
            pdf_path = state["pdf_path"]
            result = self.parser.parse(pdf_path)
            
            return {"result": result, **state}
        except Exception as e:
            logger.error(f"Error in load_pdf: {e}", exc_info=True)
    # def load_pdf(self, state: Dict) -> Dict:
    #     try:
    #         pdf_path = state["pdf_path"]
    #         loader = PyPDFLoader(pdf_path)
    #         pages = loader.load()
    #         return {"pages": pages, **state}
    #     except Exception as e:
    #         logger.error(f"Error in load_pdf: {e}", exc_info=True)

    # def split_text(self, state: Dict) -> Dict:
    #     pages = state["pages"]
    #     chunks = self.text_splitter.split_documents(pages)
    #     return {"chunks": chunks, **state}
    def split_text(self, state: Dict) -> Dict:
        result = state["result"]
        md_text = "\n\n".join(page.md for page in result.pages if page.md)
        chunks = self.text_splitter.split_text(md_text)
        return {"chunks": chunks, **state}
    
    def keyword_extraction(self, state: Dict) -> Dict:
        

        chunks = state["chunks"]
        enhanced_chunks = []

        keywords = [
            "카페인", "카페인 대사", "카페인 민감도", "섭취 권장량",
            "수면", "스트레스", "운동", "흡연", "간질환", "임신",
            "피임약 복용", "호르몬", "간기능", "고혈압", "당뇨",
            "고지혈증", "대사증후군", "BMI", "체중", "성별"
        ]

        for chunk in chunks:
            prompt = f"""
            다음 문장은 특정 유저의 카페인 섭취 통계입니다. 아래의 키워드 목록 중, 해당 문장과 관련된 키워드만 골라주세요.

            문장:
            "{chunk.page_content}"

            키워드 목록:
            {keywords}

            출력 형식:
            ["카페인", "수면", ...]
            """

            response = self.client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt
                )
            
            extracted_keywords = response.text.strip()

            

            new_metadata = chunk.metadata.copy()
            new_metadata["keywords"] = extracted_keywords

            chunk.metadata = new_metadata
            enhanced_chunks.append(chunk)

        return {"enhanced_chunks": enhanced_chunks, **state}

    def create_embeddings_and_store(self, state: Dict) -> Dict:
        try:    
            chunks = state["enhanced_chunks"]
            collection_name = state.get("collection_name", "default_collection")
            
            vectorstore = Chroma(
                embedding_function=self.embeddings,
                collection_name=collection_name,
                persist_directory="chroma_db"
            )
            
            # Always add new documents to the collection
            vectorstore.add_documents(chunks)        
            
            return {
                "status": "success",
                "collection_name": collection_name,
                "document_count": len(chunks),
            }
        except Exception as e:
            logger.error(f"Error in create_embeddings_and_store: {e}", exc_info=True)
            raise
    def retrieve_similar_chunks(self, query: str, collection_name: str = "default_collection", k: int = 3) -> list:
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory="chroma_db"
        )
        
        results = vectorstore.max_marginal_relevance_search(
        query,
        k=k*2,  # 더 많은 결과를 가져와서
        fetch_k=30,  # 더 큰 후보군에서
        lambda_mult=0.3  # 다양성에 더 큰 가중치
        )
        
        # 중복 제거 로직
        unique_results = []
        seen_content = set()
        
        for doc in results:
            # 문서 내용의 처음 100자를 기준으로 중복 체크
            content_start = doc.page_content[:100]
            if content_start not in seen_content:
                seen_content.add(content_start)
                unique_results.append(doc)
                if len(unique_results) == k:  # 원하는 개수만큼만
                    break
        
        return unique_results 
def create_pdf_processing_workflow() -> Graph:
    
    pipeline = PDFProcessingPipeline()
    
    
    workflow = Graph()
    
  
    workflow.add_node("load_pdf", pipeline.load_pdf)
    workflow.add_node("split_text", pipeline.split_text)
    workflow.add_node("keyword_extraction", pipeline.keyword_extraction)
    workflow.add_node("store_embeddings", pipeline.create_embeddings_and_store)
    
 
    workflow.add_edge("load_pdf", "split_text")
    workflow.add_edge("split_text", "keyword_extraction")
    workflow.add_edge("keyword_extraction", "store_embeddings")
    
    
    workflow.set_entry_point("load_pdf")
    workflow.set_finish_point("store_embeddings")
    
    return workflow.compile()


if __name__ == "__main__":
    pipeline = PDFProcessingPipeline()
    workflow = create_pdf_processing_workflow()
    
    
    result = workflow.invoke({
        "pdf_path": "/Users/chanwooyang/Downloads/대학생의카페인음료섭취와수면의질.pdf",
        "collection_name": "my_collection"
    })

    test_queries = [
        "카페인이 수면에 미치는 영향",
        "카페인과 졸음의 관계",
        "청소년의 카페인 섭취"
    ]
    
    for query in test_queries:
        print(f"\n=== 검색어: {query} ===")
        results = pipeline.retrieve_similar_chunks(query, "my_collection")
        for doc in results:
            print("\n문서 내용:", doc.page_content)
            print("메타데이터:", doc.metadata)