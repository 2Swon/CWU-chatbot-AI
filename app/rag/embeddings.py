from langchain.embeddings.openai import OpenAIEmbeddings
from app.config.settings import OPENAI_API_KEY, EMBEDDING_MODEL
import logging

logger = logging.getLogger(__name__)

def get_embedding_model(provider="openai"):
    """임베딩 모델 인스턴스 생성 및 반환"""
    if provider.lower() == "openai":
        logger.info(f"OpenAI 임베딩 모델 초기화: {EMBEDDING_MODEL}")
        return OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY,
            model=EMBEDDING_MODEL
        )
    else:
        raise ValueError(f"지원되지 않는 임베딩 제공자: {provider}, OpenAI만 지원됩니다")
