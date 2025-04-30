from langchain.prompts import PromptTemplate

# Cypher 생성을 위한 프롬프트 템플릿
CYPHER_GENERATION_TEMPLATE = """
Task: Neo4j 그래프 데이터베이스를 쿼리하기 위한 Cypher 문을 생성하세요.
지침: 
1. 스키마에 제공된 관계 유형과 속성만 사용하세요. 제공되지 않은 다른 관계 유형이나 속성은 사용하지 마세요.
2. RETURN 문은 반드시 쿼리의 마지막에 한 번만 사용하세요. 여러 개의 RETURN 문을 사용하지 마세요.
3. 여러 필드를 반환할 때는 뒤에 AS로 별칭을 부여하여 하나의 RETURN 문안에 콤마로 구분해서 연결하세요. 예: RETURN field1 AS Name, field2 AS Age

스키마:
{schema}

참고: 응답에 설명이나 사과를 포함하지 마세요. Cypher 문 생성 외에 다른 질문에 응답하지 마세요. 생성된 Cypher 문 외에 다른 텍스트를 포함하지 마세요.

예시:
# 인천캠퍼스의 주소가 뭐야??
MATCH (campus:Campus {{name: '인천캠퍼스'}})-[:LOCATED_AT]->(address:Address) 
RETURN address.name

# 홍성캠퍼스의 도서관 전화번호가 뭐야?
MATCH (c:Campus {{name:"홍성캠퍼스"}})-[:INCLUDES_LIBRARY]->(l:Library)-[:HAS_PHONE_NUMBER]->(pn:PhoneNumber)
RETURN pn.number AS Library_PhoneNumber

# 컴퓨터공학과에 대해 설명해줘
MATCH (d:Department {{name:"컴퓨터공학과"}})-[:HAS_DESCRIPTION]->(desc:Description) 
RETURN desc.name AS Description

# 이클래스 웹페이지 주소가 뭐야?
MATCH (:Cwuwebpage)-[:OFFERS_EClass]->(w:Webpage)
RETURN w.detail AS EClass_Webpage_Address

질문: {question}
"""

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], 
    template=CYPHER_GENERATION_TEMPLATE
)

# 최종 응답 생성을 위한 프롬프트 템플릿
QA_TEMPLATE = """너는 청운대학교 학생들을 위한 챗봇 비서야. 청운대학교와 관련된 질문에 대해 친절하게 대답해.

질문: {question}

Neo4j 쿼리 결과: {context}

쿼리 결과를 바탕으로 친절하고 정확하게 답변해. 쿼리 결과에 없는 정보는 포함하지 마. 
대답은 한국어로 해. 명백한 답변이 없는 경우 "현재 이 질문에 대한 정보가 부족합니다. 다른 질문을 해주시겠어요?"라고 말해.
"""

QA_PROMPT = PromptTemplate(
    input_variables=["question", "context"], 
    template=QA_TEMPLATE
)

# Graph RAG를 위한 프롬프트 템플릿
GRAPH_RAG_TEMPLATE = """너는 청운대학교 학생들을 위한 챗봇 비서야. 청운대학교와 관련된 질문에 대답하는데 도움을 줘.

질문: {question}

그래프 데이터베이스 정보: {graph_info}

관련 문서 컨텍스트: {document_context}

위 정보들을 바탕으로 학생의 질문에 친절하고 정확하게 답변해.
모든 대답은 한국어로 해야 해. 제공된 정보에서 찾을 수 없는 내용은 포함하지 마.
모든 답변은 사실에 기반해야 하고, 명시적인 답변이 없다면 "현재 이 질문에 대한 정보가 부족합니다. 다른 질문을 해주시겠어요?"라고 말해.
"""

GRAPH_RAG_PROMPT = PromptTemplate(
    input_variables=["question", "graph_info", "document_context"], 
    template=GRAPH_RAG_TEMPLATE
)

# 향상된 Graph RAG를 위한 프롬프트 템플릿
IMPROVED_GRAPH_RAG_TEMPLATE = """너는 청운대학교 학생들을 위한 챗봇 비서야. 청운대학교와 관련된 질문에 대답하는데 도움을 줘.

질문: {question}

그래프 데이터베이스 답변: {graph_answer}

그래프 데이터: {graph_data}

그래프 스키마 정보: {schema_info}

사용된 Cypher 쿼리: {cypher}

위 정보들을 바탕으로 학생의 질문에 친절하고 정확하게 답변해.
모든 대답은 한국어로 해야 해. 제공된 정보에서 찾을 수 없는 내용은 포함하지 마.
모든 답변은 사실에 기반해야 하고, 명시적인 답변이 없다면 "현재 이 질문에 대한 정보가 부족합니다. 다른 질문을 해주시겠어요?"라고 말해.
"""

IMPROVED_GRAPH_RAG_PROMPT = PromptTemplate(
    input_variables=["question", "graph_answer", "graph_data", "schema_info", "cypher"],
    template=IMPROVED_GRAPH_RAG_TEMPLATE
)
