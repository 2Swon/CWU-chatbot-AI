from typing import List, Dict, Any, Optional
from langchain.chains import GraphCypherQAChain
from langchain.prompts.prompt import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.graphs import Neo4jGraph
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import FAISS
from app.rag.prompt_templates import CYPHER_GENERATION_PROMPT, QA_PROMPT, GRAPH_RAG_PROMPT, IMPROVED_GRAPH_RAG_PROMPT
from app.config.settings import NEO4J_URL, NEO4J_USERNAME, NEO4J_PASSWORD, OPENAI_API_KEY, LLM_MODEL, EMBEDDING_MODEL
from app.rag.embeddings import get_embedding_model
import logging
import json

logger = logging.getLogger(__name__)

class GraphRAG:
    """그래프 기반 RAG 시스템 구현"""
    
    def __init__(self, model_provider="openai", temperature=0, embedding_provider="openai"):
        """GraphRAG 시스템 초기화"""
        self.model_provider = model_provider
        self.temperature = temperature
        self.embedding_provider = embedding_provider
        
        # Neo4j 그래프 연결
        self.graph = Neo4jGraph(
            url=NEO4J_URL, 
            username=NEO4J_USERNAME, 
            password=NEO4J_PASSWORD
        )
        
        # LLM 초기화
        self.llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model=LLM_MODEL,
            temperature=temperature
        )
        
        # 임베딩 모델 초기화
        self.embeddings = get_embedding_model(embedding_provider)
        
        # 기본 Cypher QA 체인 초기화
        self.qa_chain = GraphCypherQAChain.from_llm(
            llm=self.llm,
            graph=self.graph,
            verbose=True,
            cypher_prompt=CYPHER_GENERATION_PROMPT,
            qa_prompt=QA_PROMPT,
            allow_dangerous_requests=True  # 보안 경고 무시
        )
        
        # 벡터 저장소
        self.vector_store = None
        
        logger.info(f"GraphRAG 초기화 완료 (모델: {model_provider}, 임베딩: {embedding_provider})")
    
    def query(self, question: str) -> Dict[str, Any]:
        """기본 쿼리 메서드: Cypher 쿼리 생성 및 그래프 데이터베이스에서 응답 가져오기"""
        try:
            # Cypher QA 체인으로 쿼리 실행
            response = self.qa_chain.invoke(question)
            return {
                "answer": response["result"],
                "cypher": response["cypher"] if "cypher" in response else None,
            }
        except Exception as e:
            logger.error(f"기본 쿼리 실행 오류: {str(e)}")
            return {
                "answer": "죄송합니다. 질문을 처리하는 동안 오류가 발생했습니다.",
                "error": str(e)
            }
    
    def create_vector_store(self, documents: List[Document]) -> None:
        """문서에서 벡터 저장소 생성"""
        try:
            # 텍스트 분할기 설정
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            
            # 문서 분할
            docs = []
            for doc in documents:
                split_docs = text_splitter.split_documents([doc])
                docs.extend(split_docs)
            
            # 벡터 저장소 생성
            self.vector_store = FAISS.from_documents(docs, self.embeddings)
            logger.info(f"{len(docs)}개 문서 청크로 벡터 저장소 생성 완료")
            
        except Exception as e:
            logger.error(f"벡터 저장소 생성 오류: {str(e)}")
            raise
    
    def enhanced_graph_rag_query(self, question: str, 
                                document_sources: Optional[List[Document]] = None) -> Dict[str, Any]:
        """향상된 Graph RAG 쿼리 메서드: 그래프 데이터와 텍스트 문서를 결합하여 응답 생성"""
        try:
            # 1. 먼저 그래프 데이터에서 정보 가져오기
            graph_response = self.qa_chain.invoke(question)
            graph_info = graph_response["result"] if "result" in graph_response else ""
            
            # 2. 문서 소스가 제공된 경우, 관련 문서 컨텍스트 추출
            document_context = ""
            if document_sources and not self.vector_store:
                # 벡터 저장소가 없으면 생성
                self.create_vector_store(document_sources)
            
            # 벡터 저장소에서 관련 문서 추출
            if self.vector_store:
                retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
                relevant_docs = retriever.get_relevant_documents(question)
                document_context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            # 3. 최종 응답 생성
            final_response = self.llm.invoke(
                GRAPH_RAG_PROMPT.format(
                    question=question,
                    graph_info=graph_info,
                    document_context=document_context or "관련 문서 정보가 없습니다."
                )
            )
            
            # 결과 반환
            return {
                "answer": final_response.content,
                "graph_info": graph_info,
                "document_context": document_context,
                "cypher": graph_response.get("cypher", "")
            }
        
        except Exception as e:
            logger.error(f"향상된 Graph RAG 쿼리 실행 오류: {str(e)}")
            return {
                "answer": "죄송합니다. 질문을 처리하는 동안 오류가 발생했습니다.",
                "error": str(e)
            }
    
    def true_graph_rag_query(self, question: str) -> Dict[str, Any]:
        """GraphRAG 접근 방식을 사용한 쿼리: 그래프 탐색과 텍스트 검색 결합"""
        try:
            # 1. 질문 분석 및 그래프 기반 답변 생성
            graph_response = self.qa_chain.invoke(question)
            graph_answer = graph_response.get("result", "") 
            cypher_query = graph_response.get("cypher", "")
            
            # 2. Cypher 쿼리 실행 결과 추출
            graph_data = {}
            if cypher_query:
                try:
                    # 쿼리 유효성 검사 및 오류 수정 시도
                    # 1. 여러 RETURN 문을 하나로 합치기
                    if cypher_query.count('RETURN') > 1:
                        # 쿼리를 다시 형식화해서 하나의 RETURN으로 합침
                        lines = cypher_query.strip().split('\n')
                        non_return_lines = [line for line in lines if not line.strip().startswith('RETURN')]
                        return_lines = [line.strip() for line in lines if line.strip().startswith('RETURN')]
                        
                        # RETURN 줄에서 반환할 필드만 추출
                        return_fields = []
                        for return_line in return_lines:
                            fields = return_line.replace('RETURN', '').strip().split(',')
                            return_fields.extend([field.strip() for field in fields])
                        
                        # 새 쿼리 생성
                        fixed_query = '\n'.join(non_return_lines) + '\nRETURN ' + ', '.join(return_fields)
                        cypher_query = fixed_query
                    
                    # Neo4j 그래프 탐색 수행
                    graph_data = json.dumps(self.graph.query(cypher_query), ensure_ascii=False)[:2000]  # 너무 길면 잘라냄
                except Exception as e:
                    logger.warning(f"Cypher 쿼리 실행 오류: {str(e)}")
            
            # 3. 그래프 스키마 정보 추출
            schema_info = self.graph.schema[:1000]  # 너무 길면 잘라냄
            
            # 4. 최종 응답 생성
            final_response = self.llm.invoke(
                IMPROVED_GRAPH_RAG_PROMPT.format(
                    question=question,
                    graph_answer=graph_answer,
                    graph_data=graph_data,
                    schema_info=schema_info,
                    cypher=cypher_query
                )
            )
            
            # 결과 반환
            return {
                "answer": final_response.content,
                "graph_answer": graph_answer,
                "graph_data": graph_data,
                "schema_info": schema_info,
                "cypher": cypher_query
            }
            
        except Exception as e:
            logger.error(f"True GraphRAG 쿼리 실행 오류: {str(e)}")
            return {
                "answer": "죄송합니다. 질문을 처리하는 동안 오류가 발생했습니다.",
                "error": str(e)
            }
    
    def close(self):
        """GraphRAG 시스템 연결 종료"""
        self.graph.close()
        logger.info("GraphRAG 연결 종료")
