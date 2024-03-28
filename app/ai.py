from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
import os
app = FastAPI()

class QuestionRequest(BaseModel):
    question: str

neo4j_url = os.getenv("NEO4J_URL")
neo4j_username = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD")

graph = Neo4jGraph(url=neo4j_url, username=neo4j_username, password=neo4j_password)

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

chain = GraphCypherQAChain.from_llm(
    ChatOpenAI(temperature=0),
    graph=graph,
    verbose=False,
    cypher_prompt=CYPHER_GENERATION_PROMPT,
)

@app.post("/question")
async def get_answer(question_request: QuestionRequest):
    try:
        # 질문을 실행하고 결과를 반환합니다.
        result = chain.run(question_request.question)
        return {"answer": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")