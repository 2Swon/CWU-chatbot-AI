from typing import List, Dict, Any, Optional
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
import random
from app.config.settings import OPENAI_API_KEY, LLM_MODEL

logger = logging.getLogger(__name__)

class TestDataGenerator:
    """Test data generator for evaluation"""
    
    def __init__(self, model_name=LLM_MODEL):
        """Initialize test data generator"""
        self.llm = ChatOpenAI(api_key=OPENAI_API_KEY, model=model_name)
        logger.info("Test data generator initialized")
        
        # Prompt for generating questions
        self.question_gen_prompt = ChatPromptTemplate.from_template("""
        You are a system that generates questions about Chungwoon University.
        Please generate {num_questions} diverse questions on the following topics:
        
        - Campus locations and facilities
        - Departments and educational programs
        - Admissions and academic schedules
        - Student services and support
        - Campus life (student ID, dormitories, parking, etc.)
        
        Questions should be natural, as if asked by actual students.
        Each question should be concise and clear.
        
        Output format:
        1. [Question 1]
        2. [Question 2]
        ...
        """)
    
    def generate_questions(self, num_questions: int = 10) -> List[str]:
        """Generate evaluation questions"""
        try:
            logger.info(f"Generating {num_questions} test questions...")
            
            # Use LLM to generate questions
            response = self.llm.invoke(
                self.question_gen_prompt.format(num_questions=num_questions)
            )
            
            # Parse the response
            questions = []
            for line in response.content.strip().split('\n'):
                if line.strip() and any(line.strip().startswith(str(i) + '.') for i in range(1, num_questions + 1)):
                    question = line.strip().split('.', 1)[1].strip()
                    questions.append(question)
            
            logger.info(f"Generated {len(questions)} questions")
            return questions
        
        except Exception as e:
            logger.error(f"Error generating questions: {str(e)}")
            return []
    
    def save_questions_to_csv(self, questions: List[str], filepath: str) -> bool:
        """Save generated questions to a CSV file"""
        try:
            df = pd.DataFrame({"question": questions})
            df.to_csv(filepath, index=False, encoding='utf-8')
            logger.info(f"Questions saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving questions: {str(e)}")
            return False
    
    def load_questions_from_csv(self, filepath: str) -> List[str]:
        """Load questions from a CSV file"""
        try:
            df = pd.read_csv(filepath, encoding='utf-8')
            questions = df['question'].tolist()
            logger.info(f"Loaded {len(questions)} questions from {filepath}")
            return questions
        except Exception as e:
            logger.error(f"Error loading questions: {str(e)}")
            return []
