# 청운대학교 챗봇 AI (CWU-chatbot-AI)

이 프로젝트는 청운대학교 정보를 제공하는 그래프 기반 챗봇 시스템입니다. 두 가지 접근 방식 (LangChain과 GraphRAG)을 구현하고 비교하여 그래프 기반 RAG(Retrieval-Augmented Generation) 시스템의 성능을 향상시키는 방법을 연구합니다.

## 주요 기능

- Neo4j 그래프 데이터베이스를 활용한 지식 그래프 기반 질의응답
- LangChain을 이용한 기본 GraphCypherQAChain 구현
- GraphRAG 접근 방식을 사용한 향상된 질의응답 구현
- RAGAS 기반 자동 성능 평가 시스템
- 두 접근 방식의 성능 비교 웹 인터페이스

## 기술 스택

- **백엔드**: FastAPI, Python 3.9+
- **데이터베이스**: Neo4j (그래프 데이터베이스)
- **AI/ML**: LangChain, OpenAI API, RAGAS
- **프론트엔드**: HTML/CSS/JavaScript, Bootstrap 5, Chart.js
- **기타**: Docker (컨테이너화)

## 접근 방식 비교

1. **LangChain 접근 방식**: 
   - LangChain의 GraphCypherQAChain을 이용
   - 질문에서 Cypher 쿼리 생성 후 그래프 데이터베이스 쿼리
   - 결과를 기반으로 자연어 응답 생성

2. **GraphRAG 접근 방식**:
   - 그래프 데이터베이스 탐색과 벡터 검색 결합
   - 그래프 구조 정보와 스키마 활용
   - 관계 기반 추론 및 컨텍스트 강화

## 프로젝트 구조

```
CWU-chatbot-AI/
├── app/
│   ├── config/            # 설정 관련 코드
│   ├── evaluation/        # RAGAS 기반 평가 시스템
│   ├── graph/             # Neo4j 연결 및 그래프 관리
│   ├── rag/               # GraphRAG 구현
│   ├── static/            # 웹 인터페이스 파일
│   └── utils/             # 유틸리티 함수
├── .env                   # 환경 변수 (예: API 키)
├── .gitignore             # Git 무시 파일
├── Dockerfile             # Docker 빌드 설정
├── main.py                # 애플리케이션 진입점
├── neo4j_code.txt         # Neo4j 데이터베이스 초기화 코드
├── README.md              # 프로젝트 문서
└── requirements.txt       # Python 패키지 의존성
```

## Streamlit 웹 앱 실행

챗봇을 Streamlit 웹 인터페이스로 실행하려면 다음 단계를 따르세요:

1. 먼저 FastAPI 서버 실행:
   ```bash
   python main.py
   ```

2. 별도의 터미널에서 Streamlit 앱 실행:
   ```bash
   streamlit run streamlit_app.py
   ```
   
   또는 두 번째 터미널에서 `run_streamlit.bat` 파일을 실행합니다.

3. 웹 브라우저가 자동으로 열리면서 Streamlit 웹 앱이 표시됩니다 (일반적으로 http://localhost:8501).

### 웹 앱 기능

* 청운대학교 관련 질문하고 답변 받기
* LangChain 모델과 GraphRAG 모델 중 선택하여 사용
* 두 모델의 응답 비교 가능
* 성능 평가 탭에서 RAGAS 메트릭을 통한 모델 성능 평가 가능

## 성능 평가

RAGAS 프레임워크를 사용하여 다음 메트릭으로 성능을 평가합니다:

- **faithfulness**: 응답이 주어진 컨텍스트에 얼마나 충실한지
- **answer_relevancy**: 응답이 질문과 얼마나 관련 있는지
- **context_precision**: 검색된 컨텍스트의 정확도
- **context_recall**: 검색된 컨텍스트의 재현율
- **harmfulness**: 응답의 유해성 수준

## 기여 방법

1. 저장소 포크
2. 새 기능 브랜치 생성 (`git checkout -b feature/amazing-feature`)
3. 변경 사항 커밋 (`git commit -m 'Add some amazing feature'`)
4. 브랜치 푸시 (`git push origin feature/amazing-feature`)
5. Pull Request 오픈

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 연락처

프로젝트 관리자 - 이메일 주소

프로젝트 링크: [https://github.com/yourusername/CWU-chatbot-AI](https://github.com/yourusername/CWU-chatbot-AI)
