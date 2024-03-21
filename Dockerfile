# Python 공식 이미지를 사용. Ubuntu 기반임
FROM python:3.9-slim

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 파일 복사
COPY requirements.txt /app/

# 필요한 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 파일 복사
COPY . /app/

# FastAPI 애플리케이션을 실행하기 위한 명령어
CMD ["uvicorn", "app.ai:app", "--host", "0.0.0.0", "--port", "8000"]