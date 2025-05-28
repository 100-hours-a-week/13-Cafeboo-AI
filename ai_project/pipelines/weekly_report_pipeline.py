import sys
import os
from pathlib import Path
from google import genai
from langgraph.graph import StateGraph
import re
import warnings
from urllib3.exceptions import NotOpenSSLWarning
# OpenSSL 관련 경고 무시
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
import json

# tokenizers 병렬 처리 경고 제거
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 프로젝트 루트 경로를 찾고 sys.path에 추가
project_root = str(Path(__file__).resolve().parent.parent.parent)  # 13-Cafeboo-AI/ 디렉토리
sys.path.insert(0, project_root)

from typing import Dict, List, Any, Annotated, TypedDict, Optional, Union
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from ai_project.utils.prompt_utils import load_prompts_from_yaml
from langchain_upstage import UpstageGroundednessCheck
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
# config에서 API 키 불러오기
from ai_project.config.config import UPSTAGE_API_KEY, GOOGLE_API_KEY, MODEL_PATH

logger = logging.getLogger(__name__)

metric_to_text_model_path = "ai_project/models/HyperCLOVAX-SEED-Vision-Instruct-0.5B"
weekly_report_model_path = "ai_project/models/llama-3-Korean-Bllossom-8B"
embedding_model_path = "ai_project/models/embedding_model"



class ReportState(TypedDict, total=False):
    # 입력 데이터
    user_input: Annotated[str, "사용자의 일주일간 커피 소비 습관 및 수면 정보"]                      
    
    # 쿼리 처리 관련
    metric_narrative: Annotated[str, "수치를 자연어화 한 문장"]                   
    
    # 검색 결과 관련
    context_docs: Annotated[List[Dict], "검색된 문서 객체들"]            
    context_texts: Annotated[List[str], "검색된 문서 텍스트들"]            
    search_metadata: Annotated[Dict[str, Any], "검색 관련 메타데이터"]     
    
    # 리포트 생성 관련
    report: Annotated[str, "생성된 리포트"]                         
    
    # 그라운드니스 체크 관련
    groundedness_result: Annotated[Dict[str, Any], "그라운드니스 평가 결과"]
    
    # 최종 결과
    final_report: Annotated[str, "최종 리포트"]                   
    
    # 상태 및 오류 관리
    retry_count: Annotated[int, "재시도 횟수"]                    
    collection_name: Annotated[str, "사용할 벡터 DB 컬렉션 이름"]
    status: Annotated[str, "파이프라인 상태"]
    error: Annotated[Optional[str], "에러 메시지"]

class WeeklyReportNodes:
    def __init__(self, embedding_model, client, vectorstore , limiter):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        #self.models_loaded = False
        self.retry_limit = 3
        self.metric_to_text_prompt = load_prompts_from_yaml("ai_project/prompts/prompts.yaml")["metric_to_text_template"]
        self.report_generation_prompt = load_prompts_from_yaml("ai_project/prompts/prompts.yaml")["weekly_report_template"]
        self.embedding_model = embedding_model
        self.client = client
        self.models_loaded = True
        self.limiter = limiter
        self.vectorstore = vectorstore
    # Upstage API 키 설정
        self.upstage_api_key = UPSTAGE_API_KEY
        if not self.upstage_api_key:
            raise ValueError("UPSTAGE_API_KEY is not set in config")

        logger.info(f"WeeklyReportNodes 초기화 - 장치: {self.device}")

    # def load_models(self, state: ReportState) -> ReportState:
    #     try:
    #         logger.info("모델 로드 시작")

    #         if self.models_loaded:
    #             logger.info("이미 모델이 로드되었습니다.")
    #             state["status"] = "models_loaded"
    #             return state
            
    #         self.metric_to_text_tokenizer = AutoTokenizer.from_pretrained(
    #             metric_to_text_model_path,
    #             local_files_only=True
    #             )
    #         self.metric_to_text_model = AutoModelForCausalLM.from_pretrained(
    #             metric_to_text_model_path, 
    #             local_files_only=True,
    #             torch_dtype=torch.bfloat16
    #             )

    #         self.metric_to_text_model.to(self.device)

    #         self.weekly_report_tokenizer = AutoTokenizer.from_pretrained(weekly_report_model_path, local_files_only=True)
    #         self.weekly_report_model = AutoModelForCausalLM.from_pretrained(weekly_report_model_path, local_files_only=True, torch_dtype=torch.bfloat16)

    #         self.weekly_report_model.to(self.device)

    #         self.embeddings = HuggingFaceEmbeddings(
    #             model_name=embedding_model_path,  
    #             model_kwargs={"device": self.device}
    #         )

    #         self.models_loaded = True
    #         logger.info("모델 로드 완료")
    #         state["status"] = "models_loaded"
    #         return state
            
    #     except Exception as e:
    #         self.models_loaded = False
    #         logger.error(f"모델 로드 중 오류: {str(e)}", exc_info=True)
    #         state["status"] = "error"
    #         state["error"] = f"모델 로드 실패: {str(e)}"
    #         return state
    # def load_models(self, state: ReportState) -> ReportState:
    #     try:
    #         logger.info("모델 로드 시작")

    #         # 이미 로드된 모델이 있는지 확인
    #         if self.models_loaded:
    #             logger.info("이미 모델이 로드되었습니다.")
    #             state["status"] = "models_loaded"
    #             return state

    #         # config에서 불러온 Google API 키 사용
    #         self.client = genai.Client(api_key=GOOGLE_API_KEY)

    #         # 임베딩 모델 로드
    #         logger.info(f"임베딩 모델 로드 중: {embedding_model_path}")
    #         self.embeddings = HuggingFaceEmbeddings(
    #             model_name=embedding_model_path,  
    #             model_kwargs={"device": self.device},
    #             encode_kwargs={"normalize_embeddings": True}  # 정규화 옵션 추가
    #         )

    #         self.models_loaded = True
    #         logger.info("모델 로드 완료")
    #         state["status"] = "models_loaded"
    #         return state
            
    #     except Exception as e:
    #         self.models_loaded = False
    #         logger.error(f"모델 로드 중 오류: {str(e)}", exc_info=True)
    #         state["status"] = "error"
    #         state["error"] = f"모델 로드 실패: {str(e)}"
    #         return state
        
    # def metric_to_text(self, state: ReportState) -> ReportState:
    #     try:
    #         logger.info("Metric to Text 시작")
            
    #         # 모델 로드 확인
    #         if not self.models_loaded:
    #             logger.error("모델이 로드되지 않았습니다.")
    #             state["status"] = "error"
    #             state["error"] = "모델이 로드되지 않았습니다."
    #             return state
            
    #         # 입력 검증
    #         if "user_input" not in state or not state["user_input"]:
    #             logger.error("필수 입력 'user_input'이 없거나 비어 있습니다.")
    #             state["status"] = "error"
    #             state["error"] = "유효한 사용자 입력이 필요합니다."
    #             return state
                
    #         # 프롬프트 템플릿
    #         prompt = self.metric_to_text_prompt.format(user_input=state["user_input"])

    #         # 토큰화 및 추론
    #         inputs = self.metric_to_text_tokenizer(prompt, return_tensors="pt").to(self.device)
    #         logger.info(f"입력 토큰 길이: {inputs.input_ids.shape[1]}")

    #         # 생성 설정값
    #         max_new_tokens = 500  # 더 긴 출력 허용
    #         min_new_tokens = 50   # 최소 출력 길이 설정
            
    #         with torch.no_grad():
    #             outputs = self.metric_to_text_model.generate(
    #                 inputs.input_ids,
    #                 max_new_tokens=max_new_tokens,      
    #                 min_new_tokens=min_new_tokens,      
    #                 temperature=0.7,         
    #                 top_p=0.9,               
    #                 do_sample=True,          
    #                 pad_token_id=self.metric_to_text_tokenizer.eos_token_id  
    #             )

    #         decoded_output = self.metric_to_text_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
    #         # 응답에서 프롬프트 부분 제거
    #         if decoded_output.startswith(prompt):
    #             narrative_text = decoded_output[len(prompt):].strip()
    #         else:
    #             # 프롬프트를 찾을 수 없는 경우 전체 출력 사용
    #             narrative_text = decoded_output
            
    #         # 출력 길이 확인
    #         logger.info(f"생성된 텍스트 길이: {len(narrative_text)} 자")
    #         logger.info(f"Metric to Text 완료: {narrative_text[:100]}...")
            
    #         # state 업데이트
    #         state["metric_narrative"] = narrative_text
    #         state["status"] = "metric_processed"
            
    #         return state

    #     except Exception as e:
    #         logger.error(f"자연어 처리중 오류: {str(e)}", exc_info=True)
    #         state["status"] = "error"
    #         state["error"] = f"자연어 처리 실패: {str(e)}"
    #         return state


    async def metric_to_text(self, state: ReportState) -> ReportState:
        try:
            logger.info("Metric to Text 시작")
            
            # 모델 로드 확인
            if not self.models_loaded:
                logger.error("모델이 로드되지 않았습니다.")
                state["status"] = "error"
                state["error"] = "모델이 로드되지 않았습니다."
                return state
            
            # 입력 검증
            if "user_input" not in state or not state["user_input"]:
                logger.error("필수 입력 'user_input'이 없거나 비어 있습니다.")
                state["status"] = "error"
                state["error"] = "유효한 사용자 입력이 필요합니다."
                return state
                
            # 프롬프트 템플릿
            prompt = self.metric_to_text_prompt.format(user_input=state["user_input"])

            async with self.limiter:
                response = self.client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt
                )
            response_text = response.text
            logger.info(f"응답 텍스트: {response_text}")
            narrative_text = ""
            

            narrative_match = re.search(r'자연어:\s*(.*?)(?=키워드:|$)', response_text, re.DOTALL)
            logger.info(f"자연어 텍스트: {narrative_match}")
            if narrative_match:
                narrative_text = narrative_match.group(1)

            keyword_match = re.search(r'키워드: (.*)', response_text)
            logger.info(f"키워드 텍스트: {keyword_match}")
            if keyword_match:
                keywords = json.loads(keyword_match.group(1).strip())
            else:
                keywords = ["카페인", "수면", "건강"]
            # state 업데이트
            state["metric_narrative"] = narrative_text
            state["keywords"] = keywords
            state["status"] = "metric_processed"
            
            return state 

        except Exception as e:
            logger.error(f"자연어 처리중 오류: {str(e)}", exc_info=True)
            state["status"] = "error"
            state["error"] = f"자연어 처리 실패: {str(e)}"
            return state


    def search_documents(self, state: ReportState) -> ReportState:
        try:
            # 검색 파라미터 및 쿼리 가져오기
            query = state.get("metric_narrative", "")
            search_params = state.get("search_params", {})
            retry_count = state.get("retry_count", 0)
            strategy = state.get("search_strategy", "default")
            keywords = state.get("keywords", ["카페인", "수면", "건강"])
            
            logger.info(f"문서 검색 시작 (전략: {strategy}, 재시도 횟수: {retry_count})")
            
            # 모델 로드 확인
            if not self.models_loaded:
                logger.error("모델이 로드되지 않았습니다.")
                state["status"] = "error"
                state["error"] = "모델이 로드되지 않았습니다."
                return state
            
            # 입력 검증
            if not query:
                logger.error("검색에 사용할 쿼리가 없습니다.")
                state["status"] = "error"
                state["error"] = "검색에 사용할 자연어 쿼리가 필요합니다."
                return state
            
            # 컬렉션 이름 결정
            collection_name = state.get("collection_name", "default_collection")
            logger.info(f"사용할 컬렉션: {collection_name}")
            
            # ChromaDB 초기화
            try:
                logger.info(f"Chroma 벡터 스토어 주입 - 컬렉션: {collection_name}")
                if self.vectorstore is None:
                    vectorstore = Chroma(
                        collection_name=collection_name,
                        embedding_function=self.embedding_model,
                        persist_directory="chroma_db"
                    )
                else:
                    vectorstore = self.vectorstore
                
                # 컬렉션 내 문서 수 확인
                doc_count = vectorstore._collection.count()
                logger.info(f"컬렉션 '{collection_name}'의 문서 수: {doc_count}")
                
                if doc_count == 0:
                    logger.warning(f"컬렉션 '{collection_name}'에 문서가 없습니다.")
                    state["status"] = "search_completed"
                    state["context_docs"] = []
                    state["context_texts"] = []
                    state["search_metadata"] = {
                        "collection_name": collection_name,
                        "document_count": 0,
                        "query": query,
                        "search_strategy": strategy,
                        "retry_count": retry_count
                    }
                    return state
                
                # search_params에서 검색 파라미터 가져오기
                k = search_params.get("k_docs", 10)
                fetch_k = search_params.get("fetch_k", k * 3)
                lambda_mult = search_params.get("lambda_mult", 0.4)
                
                logger.info(f"검색 파라미터: k={k}, fetch_k={fetch_k}, lambda_mult={lambda_mult}")
                
                # 전략에 따른 검색 실행
                results = []
                
                if strategy == "metadata_keywords":
                    # 메타데이터의 keywords 필드를 기반으로 검색
                    logger.info("메타데이터 키워드 기반 검색 수행")
                    
                    # 메타데이터 필터 구성
                    # 문서 메타데이터의 keywords와 유저 keywords 간 일치도 확인
                    where_filter = None
                    
                    # 키워드 점수 계산 함수를 사용하여 컬렉션 전체 검색 후 정렬
                    all_docs = vectorstore.similarity_search(
                        query="",  # 빈 쿼리로 모든 문서 검색
                        k=doc_count if doc_count < 100 else 100,  # 적절한 제한
                        filter=where_filter
                    )
                    
                    # 키워드 일치도에 따라 문서에 점수 부여
                    scored_docs = []
                    for doc in all_docs:
                        doc_keywords = doc.metadata.get("keywords", [])
                        if not doc_keywords:
                            continue
                            
                        # 문자열인 경우 리스트로 변환
                        if isinstance(doc_keywords, str):
                            try:
                                doc_keywords = eval(doc_keywords)  # 안전하지 않을 수 있음
                            except:
                                doc_keywords = doc_keywords.split(",")
                        
                        # 키워드 일치 점수 계산
                        match_score = sum(1 for kw in keywords if kw in doc_keywords)
                        scored_docs.append((doc, match_score))
                    
                    # 점수에 따라 정렬하고 상위 k개 선택
                    scored_docs.sort(key=lambda x: x[1], reverse=True)
                    results = [doc for doc, score in scored_docs[:k]]
                    
                    logger.info(f"메타데이터 키워드 검색 결과: {len(results)}개 문서")
                
                elif strategy == "hybrid_search":
                    # 임베딩 검색과 메타데이터 검색 결합
                    logger.info("하이브리드 검색 수행 (임베딩 + 메타데이터)")
                    
                    # 임베딩 검색으로 문서 가져오기
                    embedding_results = vectorstore.max_marginal_relevance_search(
                        query,
                        k=k//2,  # 절반만 임베딩 검색
                        fetch_k=fetch_k,
                        lambda_mult=lambda_mult
                    )
                    
                    # 나머지 절반은 메타데이터 검색
                    all_docs = vectorstore.similarity_search(
                        query="",  # 빈 쿼리로 모든 문서 검색
                        k=doc_count if doc_count < 100 else 100  # 적절한 제한
                    )
                    
                    scored_docs = []
                    for doc in all_docs:
                        doc_keywords = doc.metadata.get("keywords", [])
                        if not doc_keywords:
                            continue
                            
                        # 문자열인 경우 리스트로 변환
                        if isinstance(doc_keywords, str):
                            try:
                                doc_keywords = eval(doc_keywords)
                            except:
                                doc_keywords = doc_keywords.split(",")
                        
                        # 키워드 일치 점수 계산
                        match_score = sum(1 for kw in keywords if kw in doc_keywords)
                        scored_docs.append((doc, match_score))
                    
                    # 점수에 따라 정렬하고 상위 k/2개 선택
                    scored_docs.sort(key=lambda x: x[1], reverse=True)
                    metadata_results = [doc for doc, score in scored_docs[:k//2]]
                    
                    # 두 결과 합치기 (중복 제거)
                    seen_ids = set()
                    results = []
                    
                    for doc in embedding_results + metadata_results:
                        doc_id = doc.metadata.get("id", "")
                        if doc_id and doc_id in seen_ids:
                            continue
                        results.append(doc)
                        if doc_id:
                            seen_ids.add(doc_id)
                    
                    logger.info(f"하이브리드 검색 결과: {len(results)}개 문서")
                
                else:  # "embedding_only" 또는 기본
                    # 기본 임베딩 검색
                    logger.info("임베딩 기반 검색 수행")
                    results = vectorstore.max_marginal_relevance_search(
                        query,
                        k=k,
                        fetch_k=fetch_k,
                        lambda_mult=lambda_mult
                    )
                    
                    logger.info(f"임베딩 검색 결과: {len(results)}개 문서")
                
                logger.info(f"검색 완료: {len(results)}개 문서 검색됨")
                
                # 검색 결과 처리
                context_docs = []
                context_texts = []
                
                for i, doc in enumerate(results):
                    # 문서 내용 디버깅 (첫 100자만)
                    truncated_content = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                    logger.debug(f"문서 {i+1}: {truncated_content}")
                    
                    # 문서 객체와 텍스트 저장
                    context_docs.append({
                        "page_content": doc.page_content,
                        "metadata": doc.metadata
                    })
                    context_texts.append(doc.page_content)
                
                # 결과 로깅
                logger.info(f"검색된 문서 수: {len(context_docs)}")
                if context_docs:
                    logger.info(f"첫 번째 문서 샘플: {context_texts[0][:100]}...")
                
                # 그라운드니스 체크에서 사용할 수 있도록 문자열 형태로 저장
                state["search_results"] = "\n\n---\n\n".join(context_texts)
                
                # state 업데이트
                state["context_docs"] = context_docs
                state["context_texts"] = context_texts
                state["search_metadata"] = {
                    "collection_name": collection_name,
                    "document_count": len(context_docs),
                    "query": query,
                    "search_strategy": strategy,
                    "retry_count": retry_count,
                    "search_time": "현재 시간"
                }
                state["status"] = "search_completed"
                
                return state
                
            except Exception as e:
                logger.error(f"벡터 스토어 검색 중 오류: {str(e)}", exc_info=True)
                state["status"] = "error"
                state["error"] = f"문서 검색 실패: {str(e)}"
                return state
            
        except Exception as e:
            logger.error(f"문서 검색 중 오류: {str(e)}", exc_info=True)
            state["status"] = "error"
            state["error"] = f"문서 검색 실패: {str(e)}"
            return state
    
    # def generate_report(self, state: ReportState) -> ReportState:
    #     """
    #     검색된 문서와 쿼리를 바탕으로 종합 리포트를 생성합니다.
    #     """
    #     try:
    #         logger.info("리포트 생성 시작")
            
            
    #         if not self.models_loaded:
    #             logger.error("모델이 로드되지 않았습니다.")
    #             state["status"] = "error"
    #             state["error"] = "모델이 로드되지 않았습니다."
    #             return state
            
            
    #         if "metric_narrative" not in state or not state["metric_narrative"]:
    #             logger.error("필수 입력 'metric_narrative'가 없거나 비어 있습니다.")
    #             state["status"] = "error"
    #             state["error"] = "리포트 생성을 위한 자연어 쿼리가 필요합니다."
    #             return state
            
    #         # 문서 존재 여부 확인
    #         if "context_texts" not in state or not state["context_texts"]:
    #             logger.warning("검색된 문서가 없습니다. 제한된 정보로 리포트를 생성합니다.")
    #             context_texts = ["관련 정보를 찾을 수 없습니다."]
    #         else:
    #             context_texts = state["context_texts"]
            
    #         # 문서 텍스트 결합 (최대 길이 제한)
    #         combined_text = "\n\n---\n\n".join(context_texts)
            
            
    #         # 로드된 프롬프트 사용
    #         prompt = self.report_generation_prompt.format(
    #             metric_narrative=state["metric_narrative"],
    #             combined_text=combined_text
    #         )
            
    #         # 토큰화 및 추론
    #         inputs = self.weekly_report_tokenizer(prompt, return_tensors="pt").to(self.device)
    #         logger.info(f"입력 토큰 길이: {inputs.input_ids.shape[1]}")
            
    #         # 생성 설정값
    #         max_new_tokens = 500  
    #         min_new_tokens = 300   # 최소 출력 길이 설정
            
    #         logger.info("리포트 생성 중...")
    #         with torch.no_grad():
    #             outputs = self.weekly_report_model.generate(
    #                 inputs.input_ids,
    #                 max_new_tokens=max_new_tokens,      
    #                 min_new_tokens=min_new_tokens,      
    #                 temperature=0.6,         
    #                 top_p=0.9,               
    #                 do_sample=True,          
    #                 pad_token_id=self.weekly_report_tokenizer.eos_token_id  
    #             )
            
    #         decoded_output = self.weekly_report_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
    #         # 응답에서 프롬프트 부분 제거
    #         if decoded_output.startswith(prompt):
    #             report_text = decoded_output[len(prompt):].strip()
    #         else:
    #             # 프롬프트를 찾을 수 없는 경우 전체 출력 사용
    #             report_text = decoded_output
            
    #         # 출력 길이 확인
    #         logger.info(f"생성된 리포트 길이: {len(report_text)} 자")
    #         logger.info(f"리포트 생성 완료: {report_text[:150]}...")
            
    #         # state 업데이트
    #         state["report"] = report_text
    #         state["status"] = "report_generated"
            
    #         return state
        
    #     except Exception as e:
    #         logger.error(f"리포트 생성 중 오류: {str(e)}", exc_info=True)
    #         state["status"] = "error"
    #         state["error"] = f"리포트 생성 실패: {str(e)}"
    #         return state
    async def generate_report(self, state: ReportState) -> ReportState:
        """
        검색된 문서와 쿼리를 바탕으로 종합 리포트를 생성합니다.
        """
        try:
            logger.info("리포트 생성 시작")
            
            
            if not self.models_loaded:
                logger.error("모델이 로드되지 않았습니다.")
                state["status"] = "error"
                state["error"] = "모델이 로드되지 않았습니다."
                return state
            
            
            if "metric_narrative" not in state or not state["metric_narrative"]:
                logger.error("필수 입력 'metric_narrative'가 없거나 비어 있습니다.")
                state["status"] = "error"
                state["error"] = "리포트 생성을 위한 자연어 쿼리가 필요합니다."
                return state
            
            # 문서 존재 여부 확인
            if "context_texts" not in state or not state["context_texts"]:
                logger.warning("검색된 문서가 없습니다. 제한된 정보로 리포트를 생성합니다.")
                context_texts = ["관련 정보를 찾을 수 없습니다."]
            else:
                context_texts = state["context_texts"]
            
            # 문서 텍스트 결합 (최대 길이 제한)
            combined_text = "\n\n---\n\n".join(context_texts)
            
            
            # 로드된 프롬프트 사용
            prompt = self.report_generation_prompt.format(
                metric_narrative=state["metric_narrative"],
                combined_text=combined_text
            )
            async with self.limiter:
                response = self.client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt
                )
            report_text = response.text
            
            # 출력 길이 확인
            logger.info(f"생성된 리포트 길이: {len(report_text)} 자")
            logger.info(f"리포트 생성 완료: {report_text[:150]}...")
            
            # state 업데이트
            state["report"] = report_text
            state["status"] = "report_generated"
            
            return state
        
        except Exception as e:
            logger.error(f"리포트 생성 중 오류: {str(e)}", exc_info=True)
            state["status"] = "error"
            state["error"] = f"리포트 생성 실패: {str(e)}"
            return state


    async def check_groundedness(self, state: ReportState) -> ReportState:
        try:
            # Upstage Groundedness Check 초기화
            checker = UpstageGroundednessCheck()
            
            # 리포트와 컨텍스트 준비
            report = state["report"]
            
            # search_results에서 컨텍스트 가져오기 (문자열로 저장됨)
            context = state.get("search_results", "")
            
            # 컨텍스트가 비어 있는 경우 처리
            if not context or len(context) < 10:
                # context_texts에서 직접 구성
                if "context_texts" in state and state["context_texts"]:
                    context = "\n\n---\n\n".join(state["context_texts"])
                else:
                    # 기본 컨텍스트 제공
                    logger.warning("컨텍스트가 비어 있어 기본 컨텍스트를 사용합니다.")
                    context = "카페인은 커피, 차, 에너지 드링크 등에 함유된 각성 물질입니다. 일일 권장 섭취량은 400mg 이하입니다."
            
            # 그라운드니스 체크 수행
            logger.info("그라운드니스 체크 수행 중...")

            async with self.limiter:
                groundedness_result = checker.invoke({
                    "context": context,
                    "answer": report
                })
            
            # 응답 형식 수정
            groundedness_status = groundedness_result  # 직접 문자열로 반환되는 것으로 보임
            logger.info(f"그라운드니스 결과: {groundedness_status}")
            
            state["groundedness_result"] = {
                "status": groundedness_status,
                "is_grounded": groundedness_status.lower() == "grounded",
                "details": groundedness_result
            }
            
            # 로그 기록
            if groundedness_status.lower() == "grounded":
                logger.info("리포트가 참고 문서에 기반하고 있습니다.")
            elif groundedness_status.lower() == "notgrounded":
                logger.warning("리포트가 참고 문서에 충분히 기반하지 않습니다.")
            else:  # notSure
                logger.warning("리포트의 그라운드니스 상태를 확실히 판단할 수 없습니다.")
            
            state["status"] = "groundedness_checked"
            return state
            
        except Exception as e:
            logger.error(f"그라운드니스 검사 중 오류: {str(e)}")
            state["groundedness_result"] = {
                "status": "notSure",
                "is_grounded": False,
                "error": str(e)
            }
            state["status"] = "groundedness_check_failed"
            return state
    
    def should_retry_search(self, state: ReportState) -> str:
        """
        그라운드니스 점수에 따라 재시도 여부 결정
        """
        # 재시도 횟수 확인
        retry_count = state.get("retry_count", 0)
        if retry_count >= self.retry_limit:
            logger.warning(f"최대 재시도 횟수({self.retry_limit})에 도달했습니다.")
            return "finalize_report"
        
        # 그라운드니스 상태 확인
        groundedness_result = state.get("groundedness_result", {})
        groundedness_status = groundedness_result.get("status", "notSure").lower()
        
        logger.info(f"그라운드니스 상태 확인: {groundedness_status} (재시도 {retry_count}/{self.retry_limit})")
        
        # 현재 검색 전략 확인
        current_strategy = state.get("search_strategy", "embedding_only")
        
        # 그라운드니스 결과가 좋다면 최종화
        if groundedness_status == "grounded":
            logger.info("리포트가 충분히 그라운딩되어 있어 최종화 단계로 진행합니다.")
            return "finalize_report"
        
        # 이미 모든 전략을 시도했는지 확인
        if retry_count >= 2:  # 3가지 전략을 모두 시도한 경우
            logger.warning("모든 검색 전략을 시도했으나 그라운딩이 불충분합니다.")
            return "finalize_report"
        
        # 그라운드니스가 충분하지 않다면 다음 전략으로
        logger.info(f"리포트가 충분히 그라운딩되어 있지 않아 다음 검색 전략을 시도합니다. (현재 전략: {current_strategy})")
        return "select_strategy"
    
    def finalize_report(self, state: ReportState) -> ReportState:
        """
        최종 보고서를 생성합니다.
        """
        try:
            logger.info("보고서 최종화 시작")
            
            if "report" not in state or not state["report"]:
                logger.error("필수 입력 'report'가 없거나 비어 있습니다.")
                state["status"] = "error"
                state["error"] = "최종화할 리포트가 없습니다."
                return state
            
            report_text = state["report"]
             #제목(#) 제거
            report_text = re.sub(r'#+\s+', '', report_text)
             #굵은 글씨(**) 제거
            report_text = re.sub(r'\*\*(.*?)\*\*', r'\1', report_text)
            #  기울임체(*) 제거
            # report_text = re.sub(r'\*(.*?)\*', r'\1', report_text)
        
            # 최종 보고서 생성
            final_report = report_text
            state["final_report"] = final_report
            
            if "groundedness_result" in state and state["groundedness_result"].get("is_grounded", False):
                state["status"] = "completed"
                logger.info("보고서가 신뢰할 수 있는 자료에 기반하여 완료되었습니다.")
            else:
                status_detail = state.get("groundedness_result", {}).get("status", "unknown")
                state["status"] = f"completed_with_warning_{status_detail}"
                logger.warning(f"보고서가 완료되었으나 신뢰도에 주의가 필요합니다. (상태: {status_detail})")
            
            logger.info("보고서 최종화 완료")
            return state
            
        except Exception as e:
            logger.error(f"리포트 최종화 중 오류: {str(e)}", exc_info=True)
            state["status"] = "error"
            state["error"] = f"리포트 최종화 실패: {str(e)}"
            return state

    def select_search_strategy(self, state: ReportState) -> ReportState:
        try:
            # 재시도 횟수 확인 및 증가
            retry_count = state.get("retry_count", 0)
            state["retry_count"] = retry_count + 1
            
            
            # 재시도 횟수에 따라 순차적으로 전략 선택
            strategies = ["embedding_only", "metadata_keywords", "hybrid_search"]
            
            # 재시도 횟수가 전략 수를 초과하면 마지막 전략 사용
            strategy_index = min(retry_count, len(strategies) - 1)
            selected_strategy = strategies[strategy_index]
            
            logger.info(f"전략 선택: {selected_strategy} (재시도 {retry_count+1}/{self.retry_limit})")
            
            # 선택된 전략을 state에 저장
            state["search_strategy"] = selected_strategy
            state["status"] = "strategy_selected"
            
            return state
            
        except Exception as e:
            logger.error(f"검색 전략 선택 중 오류: {str(e)}")
            # 오류 발생 시 기본 전략 사용
            state["search_strategy"] = "embedding_only"
            state["status"] = "strategy_selection_error"
            return state

    def reconstruct_query(self, state: ReportState) -> ReportState:
        try:
            # 선택된 전략과 원래 쿼리 확인
            strategy = state.get("search_strategy", "embedding_only")
            original_query = state.get("metric_narrative", "")
            retry_count = state.get("retry_count", 0)
            
            logger.info(f"쿼리 재구성 시작 (전략: {strategy})")
            
            # 전략에 따른 검색 파라미터 설정 (쿼리 자체는 변경하지 않음)
            if strategy == "embedding_only":
                # 임베딩 검색을 위한 파라미터
                search_params = {
                    "k_docs": 8,
                    "fetch_k": 25,
                    "lambda_mult": 0.4  # 다양성과 관련성의 균형
                }
                
            elif strategy == "metadata_keywords":
                # 메타데이터 키워드 검색을 위한 파라미터
                search_params = {
                    "k_docs": 10,
                    "fetch_k": 30,
                    "lambda_mult": 0.5  # 더 높은 관련성 강조
                }
                
            elif strategy == "hybrid_search":
                # 하이브리드 검색을 위한 파라미터
                search_params = {
                    "k_docs": 12,
                    "fetch_k": 35,
                    "lambda_mult": 0.35  # 다양성 강조
                }
                
            else:
                # 기본 파라미터
                search_params = {
                    "k_docs": 10,
                    "fetch_k": 30,
                    "lambda_mult": 0.4
                }
            
            # 쿼리는 변경하지 않고 검색 파라미터만 저장
            state["search_params"] = search_params
            
            logger.info(f"현재 쿼리 유지: {original_query[:50]}...")
            logger.info(f"검색 파라미터: {search_params}")
            
            state["status"] = "query_reconstructed"
            return state
            
        except Exception as e:
            logger.error(f"쿼리 재구성 중 오류: {str(e)}")
            # 오류 발생 시 원래 쿼리 유지, 기본 파라미터 설정
            state["search_params"] = {
                "k_docs": 10,
                "fetch_k": 30,
                "lambda_mult": 0.4
            }
            state["status"] = "query_reconstruction_error"
            return state


class WeeklyReportPipeline:
    """
    주간 리포트 생성을 위한 LangGraph 기반 파이프라인 클래스
    """
    def __init__(self, embedding_model, client, vectorstore, limiter):
        """
        파이프라인 초기화 및 그래프 구성
        """
        #외부에서 임베딩 모델이랑 클라이언트 주입
        self.nodes = WeeklyReportNodes(embedding_model, client, vectorstore, limiter)
        self.graph = self._build_graph()
        logger.info("주간 리포트 파이프라인이 초기화되었습니다.")
        
    def _build_graph(self) -> StateGraph:
        """
        LangGraph 그래프를 구성합니다.
        """
        # 상태 그래프 빌더 생성
        builder = StateGraph(ReportState)
        
        # 노드 추가
        #builder.add_node("load_models", self.nodes.load_models)
        builder.add_node("metric_to_text", self.nodes.metric_to_text)
        builder.add_node("search_documents", self.nodes.search_documents)
        builder.add_node("generate_report", self.nodes.generate_report)
        builder.add_node("check_groundedness", self.nodes.check_groundedness)
        builder.add_node("finalize_report", self.nodes.finalize_report)
        builder.add_node("select_strategy", self.nodes.select_search_strategy)
        builder.add_node("reconstruct_query", self.nodes.reconstruct_query)
        
        # 시작점 설정
        builder.set_entry_point("metric_to_text")
        
        # 기본 흐름 엣지 추가
        #builder.add_edge("load_models", "metric_to_text")
        builder.add_edge("metric_to_text", "search_documents")
        builder.add_edge("search_documents", "generate_report")
        builder.add_edge("generate_report", "check_groundedness")
        
        # 조건부 라우팅
        builder.add_conditional_edges(
            "check_groundedness",
            self.nodes.should_retry_search,
            {
                "select_strategy": "select_strategy",
                "finalize_report": "finalize_report"
            }
        )
        
        # 전략 선택 후 쿼리 재구성
        builder.add_edge("select_strategy", "reconstruct_query")
        
        # 쿼리 재구성 후 다시 검색
        builder.add_edge("reconstruct_query", "search_documents")
        
        
        
        # 종료 노드 설정
        builder.add_edge("finalize_report", END)
        
        # 그래프 컴파일
        return builder.compile()
    
    async def run(self, input_data: dict) -> dict:
        """
        파이프라인을 실행합니다.
        
        Args:
            input_data: 입력 데이터 사전, 최소한 'user_input' 키를 포함해야 함
            
        Returns:
            dict: 파이프라인 실행 결과
        """
        try:
            # 입력 검증
            if "user_input" not in input_data or not input_data["user_input"]:
                raise ValueError("유효한 'user_input'이 필요합니다.")
            
            # 초기 상태 생성
            initial_state: ReportState = {
                "status": "started",
                "retry_count": 0
            }
            
            # user_input이 사전인 경우 JSON 문자열로 변환
            user_input = input_data["user_input"]
            if isinstance(user_input, dict):
                import json
                initial_state["user_input"] = json.dumps(user_input, ensure_ascii=False, indent=2)
            else:
                initial_state["user_input"] = user_input
            
            # 컬렉션 이름이 제공된 경우 추가
            
            initial_state["collection_name"] = "default_collection"
            
            logger.info(f"파이프라인 실행 시작: {initial_state}")
            
            # 그래프 실행
            result = await self.graph.ainvoke(initial_state)
            
            if result["status"] == "completed" or result["status"] == "completed_with_warning":
                logger.info("파이프라인 실행 완료")
            else:
                logger.warning(f"파이프라인 비정상 종료: {result['status']}")
                
            return result
            
        except Exception as e:
            logger.error(f"파이프라인 실행 중 오류: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error": f"파이프라인 실행 실패: {str(e)}"
            }


if __name__ == "__main__":
    import os
    from google import genai
    from langchain_huggingface import HuggingFaceEmbeddings
    import asyncio
    from langchain_chroma import Chroma
    from aiolimiter import AsyncLimiter
    async def run_test():
        # 임베딩 모델, 클라이언트 생성 후 주입
        embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_path,
            model_kwargs={"device": "cpu"}
        )
        client = genai.Client(api_key="AIzaSyB5fcVPmkegjZ1dBe0Yy4spgplhVX5B-D8")
        vectorstore = Chroma(
            collection_name="default_collection",
            embedding_function=embedding_model,
            persist_directory="chroma_db"
        )
        limiter = AsyncLimiter(10, 60)
        pipeline = WeeklyReportPipeline(embedding_model=embedding_model, client=client, vectorstore=vectorstore, limiter=limiter)
        coffee_sleep_data = {
          "user_id": "peter",
          "period": "2025-05-12 ~ 2025-05-18",
          "avg_caffeine_per_day": 234.28572,
          "recommended_daily_limit": 400.0,
          "percentage_of_limit": 58.57143,
          "highlight_day_high": "TUE",
          "highlight_day_low": "WED",
          "first_coffee_avg": "13:26",
          "last_coffee_avg": "17:42",
          "late_night_caffeine_days": 3,
          "over_100mg_before_sleep_days": 1,
          "average_sleep_quality": "good"
        }
        
        
        result = await pipeline.run({"user_input": coffee_sleep_data, "collection_name": "default_collection"})
        if "final_report" in result:
            print(repr(result["final_report"]))
        else:
            print(f"오류: {result.get('error', '알 수 없는 오류')}")
    
    # 비동기 함수 실행
    asyncio.run(run_test())