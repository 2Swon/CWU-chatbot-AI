from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from langchain.chains import GraphCypherQAChain
from langchain.graphs import Neo4jGraph
from langchain.chat_models import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from app.graph.neo4j_manager import Neo4jManager
from app.rag.graph_rag import GraphRAG
from app.evaluation.evaluation_system import EvaluationSystem, sanitize_float_values, JSONEncoder
import os
import json
import logging
from dotenv import load_dotenv

# 커스텀 응답 클래스
# FastAPI의 기본 응답 대신 사용하여 특수 부동 소수점 값 처리
class SafeJSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        sanitized_content = sanitize_float_values(content)
        return json.dumps(
            sanitized_content,
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
            cls=JSONEncoder,
        ).encode("utf-8")

# .env 파일에서 환경 변수 로드
load_dotenv()


# OpenAI 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
# 경고 메시지 어댑하기
import warnings
from langchain_core._api.deprecation import LangChainDeprecationWarning
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning, module="langchain")
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning, module="langchain_community")
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning, module="ragas")

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# FastAPI 앱 초기화
app = FastAPI(title="CWU Chatbot API", description="청운대학교 챗봇 AI API")

# 정적 파일 마운트
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# 템플릿 설정
templates = Jinja2Templates(directory="app/static")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 배포 시에는 특정 출처만 허용하도록 변경
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Neo4j 환경 변수 로드
neo4j_url = os.getenv("NEO4J_URI", "neo4j+s://da08dacd.databases.neo4j.io")
neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
neo4j_password = os.getenv("NEO4J_PASSWORD", "Gcmf4Gl4vbbO9g-W7Xno03Ujpaq5TrC2bgt4nvAlBRw")

# 모델 초기화
graph = Neo4jGraph(url=neo4j_url, username=neo4j_username, password=neo4j_password)
neo4j_manager = Neo4jManager()

# LangChain 기반 모델 초기화
CYPHER_GENERATION_TEMPLATE = """
Task: Generate Cypher statement to query a graph database.
Instructions: Use only the provided relationship types and properties in the schema. Do not use any other relationship types or properties that are not provided.
Schema: {schema}
Note: Do not include any explanations or apologies in your responses. Do not respond to any questions that might ask anything else than for you to construct a Cypher statement. Do not include any text except the generated Cypher statement.
Examples: Here are a few examples of generated Cypher statements for particular questions:
# 인천캠퍼스의 주소가 뭐야??
MATCH (campus:Campus {{name: '인천캠퍼스'}})-[:LOCATED_AT]->(address:Address) 
RETURN address.name

The question is: {question}
"""

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
)

langchain_model = GraphCypherQAChain.from_llm(
    ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY),
    graph=graph,
    verbose=False,
    cypher_prompt=CYPHER_GENERATION_PROMPT,
    allow_dangerous_requests=True  # 보안 경고 무시
)

# GraphRAG 모델 초기화
graph_rag_model = GraphRAG(model_provider="openai", temperature=0)

# 평가 시스템 초기화
evaluation_system = None  # 필요할 때 초기화

# 질문 요청 모델
class QuestionRequest(BaseModel):
    question: str
    mode: Optional[str] = "langchain"  # 'langchain' 또는 'graphrag'

# 평가 요청 모델
class EvaluationRequest(BaseModel):
    questions: Optional[List[str]] = None
    sample_count: Optional[int] = 10

# 결과 비교 모델
class ComparisonRequest(BaseModel):
    question: str

# 경로 정의
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api")
async def api_root():
    return SafeJSONResponse({"message": "청운대학교 챗봇 API에 오신 것을 환영합니다!"})

@app.post("/question")
async def get_answer(question_request: QuestionRequest):
    try:
        question = question_request.question
        mode = question_request.mode.lower()
        
        if mode == "langchain":
            # LangChain 모델 사용
            result = langchain_model.run(question)
            return SafeJSONResponse({"answer": result, "mode": "langchain"})
        
        elif mode == "graphrag":
            # GraphRAG 모델 사용
            result = graph_rag_model.true_graph_rag_query(question)
            return SafeJSONResponse({"answer": result["answer"], "mode": "graphrag"})
        
        else:
            raise HTTPException(status_code=400, detail=f"지원되지 않는 모드: {mode}. 'langchain' 또는 'graphrag'를 사용하세요.")
    
    except Exception as e:
        logger.error(f"질문 처리 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"질문 처리 중 오류가 발생했습니다: {str(e)}")

@app.post("/compare")
async def compare_answers(comparison_request: ComparisonRequest):
    try:
        question = comparison_request.question
        
        # 두 모델에서 응답 가져오기
        langchain_result = langchain_model.run(question)
        graphrag_result = graph_rag_model.true_graph_rag_query(question)
        
        return SafeJSONResponse({
            "question": question,
            "langchain_answer": langchain_result,
            "graphrag_answer": graphrag_result["answer"],
        })
    
    except Exception as e:
        logger.error(f"비교 처리 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"비교 처리 중 오류가 발생했습니다: {str(e)}")

@app.post("/evaluate")
async def evaluate_systems(evaluation_request: EvaluationRequest):
    try:
        global evaluation_system
        
        # 평가 시스템이 초기화되지 않았다면 초기화
        if evaluation_system is None:
            evaluation_system = EvaluationSystem(
                langchain_model, 
                graph_rag_model,
                "LangChain",
                "GraphRAG"
            )
        
        # 평가 실행
        results = evaluation_system.run_evaluation(
            questions=evaluation_request.questions,
            sample_count=evaluation_request.sample_count
        )
        
        # 결과를 직접 JSONResponse로 제공하여 JSON 인코딩 문제 해결
        sanitized_results = sanitize_float_values(results)
        return JSONResponse(
            content={"results": sanitized_results},
            media_type="application/json"
        )
    
    except Exception as e:
        logger.error(f"평가 처리 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"평가 처리 중 오류가 발생했습니다: {str(e)}")

@app.get("/schema")
async def get_schema():
    try:
        schema = neo4j_manager.get_detailed_schema()
        return SafeJSONResponse({"schema": schema})
    except Exception as e:
        logger.error(f"스키마 가져오기 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"스키마 가져오기 중 오류가 발생했습니다: {str(e)}")

# 종료 이벤트 핸들러
@app.on_event("shutdown")
async def shutdown_event():
    try:
        neo4j_manager.close()
        graph_rag_model.close()
        logger.info("모든 연결 종료 완료")
    except Exception as e:
        logger.error(f"서버 종료 중 오류: {str(e)}")
