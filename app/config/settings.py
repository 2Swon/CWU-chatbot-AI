import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

# Neo4j 설정
NEO4J_URL = os.getenv("NEO4J_URI", "neo4j+s://da08dacd.databases.neo4j.io")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
AURA_INSTANCEID = os.getenv("AURA_INSTANCEID", "da08dacd")
AURA_INSTANCENAME = os.getenv("AURA_INSTANCENAME", "Instance01")

# OpenAI 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")

# 로깅 설정
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# 캐싱 설정
CACHE_EMBEDDINGS = os.getenv("CACHE_EMBEDDINGS", "True").lower() in ("true", "1", "t")

# RAGAS 평가 설정
RAGAS_EVAL_ENABLED = os.getenv("RAGAS_EVAL_ENABLED", "True").lower() in ("true", "1", "t")
RAGAS_SAMPLE_COUNT = int(os.getenv("RAGAS_SAMPLE_COUNT", "10"))
