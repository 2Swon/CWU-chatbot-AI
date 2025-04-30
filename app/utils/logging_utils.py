import logging
import os
import sys
from datetime import datetime
from pathlib import Path

def setup_logging(log_level=logging.INFO, log_to_file=True):
    """
    로깅 설정을 초기화합니다.
    
    Args:
        log_level: 로깅 레벨 (기본값: logging.INFO)
        log_to_file: 파일에 로그를 기록할지 여부 (기본값: True)
    """
    # 프로젝트 루트 디렉토리 찾기
    root_dir = Path(__file__).parent.parent.parent
    logs_dir = root_dir / "logs"
    
    # 로그 디렉토리가 없으면 생성
    if log_to_file and not logs_dir.exists():
        os.makedirs(logs_dir, exist_ok=True)
    
    # 기본 로깅 설정
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    # 파일 로깅 추가
    if log_to_file:
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = logs_dir / f"cwu_chatbot_{today}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)
    
    logger = logging.getLogger(__name__)
    logger.info(f"로깅 시스템 초기화 완료 (레벨: {logging.getLevelName(log_level)})")

def get_logger(name):
    """
    지정된 이름으로 로거를 반환합니다.
    
    Args:
        name: 로거 이름 (일반적으로 __name__ 사용)
    
    Returns:
        Logger: 구성된 로거 객체
    """
    return logging.getLogger(name)

class LoggerMixin:
    """
    클래스에 로깅 기능을 추가하는 믹스인
    """
    
    @property
    def logger(self):
        """클래스 이름을 기반으로 로거 반환"""
        if not hasattr(self, "_logger"):
            self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        return self._logger
