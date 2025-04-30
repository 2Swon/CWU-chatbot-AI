from typing import List, Dict, Any, Optional
from langchain.graphs import Neo4jGraph
from app.config.settings import NEO4J_URL, NEO4J_USERNAME, NEO4J_PASSWORD, AURA_INSTANCEID
import logging

logger = logging.getLogger(__name__)

class Neo4jManager:
    """Neo4j 데이터베이스 관리 클래스"""
    
    def __init__(self):
        """Neo4j 연결 초기화"""
        self.graph = Neo4jGraph(
            url=NEO4J_URL,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD
        )
        logger.info("Neo4j 연결 설정 완료")
        
    def get_schema(self) -> str:
        """Neo4j 데이터베이스 스키마 추출"""
        return self.graph.schema
        
    def execute_cypher(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Cypher 쿼리 실행 및 결과 반환"""
        try:
            result = self.graph.query(query, params=params or {})
            return result
        except Exception as e:
            logger.error(f"Cypher 쿼리 실행 오류: {str(e)}")
            raise
    
    def get_node_properties(self, labels: List[str]) -> Dict[str, List[str]]:
        """지정된 레이블을 가진 노드의 모든 속성 가져오기"""
        properties = {}
        for label in labels:
            query = f"""
            MATCH (n:{label})
            WITH keys(n) AS keys
            UNWIND keys AS key
            RETURN DISTINCT key
            """
            result = self.execute_cypher(query)
            properties[label] = [record["key"] for record in result]
        return properties
    
    def get_relationship_types(self) -> List[str]:
        """데이터베이스의 모든 관계 유형 가져오기"""
        query = """
        CALL db.relationshipTypes() YIELD relationshipType
        RETURN DISTINCT relationshipType
        """
        result = self.execute_cypher(query)
        return [record["relationshipType"] for record in result]
    
    def get_node_labels(self) -> List[str]:
        """데이터베이스의 모든 노드 레이블 가져오기"""
        query = """
        CALL db.labels() YIELD label
        RETURN DISTINCT label
        """
        result = self.execute_cypher(query)
        return [record["label"] for record in result]
    
    def get_detailed_schema(self) -> str:
        """Neo4j 데이터베이스의 상세 스키마 정보 생성"""
        labels = self.get_node_labels()
        relationships = self.get_relationship_types()
        properties = self.get_node_properties(labels)
        
        schema_text = "Node Labels:\n"
        for label in labels:
            schema_text += f"- {label}\n"
            if label in properties:
                schema_text += "  Properties:\n"
                for prop in properties[label]:
                    schema_text += f"  - {prop}\n"
        
        schema_text += "\nRelationship Types:\n"
        for rel in relationships:
            schema_text += f"- {rel}\n"
            
        return schema_text
    
    def close(self):
        """Neo4j 연결 종료"""
        self.graph.close()
        logger.info("Neo4j 연결 종료")
