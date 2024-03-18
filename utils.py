from loguru import logger # streamlit 구동기록을 로그로 남기기 위해
import tiktoken # 토큰 개수를 세기위해
from langchain.chains import ConversationalRetrievalChain # 메모리를 가진 chain을 사용하기 위해
from langchain.chat_models import ChatOpenAI # openai llm

from langchain.document_loaders import PyPDFLoader # pdf 파일
from langchain.document_loaders import Docx2txtLoader #docx 파일
from langchain.document_loaders import UnstructuredPowerPointLoader # 여러 유형의 문서들을 넣어도 이해 가능하도록 만들기 위해

from langchain.text_splitter import RecursiveCharacterTextSplitter # text를 나누기 위해
from langchain.embeddings import HuggingFaceEmbeddings # 한국어에 특화된 embedding model을 불러오기 위해

from langchain.memory import ConversationBufferMemory # 설정한 개수만큼 대화를 메모리에 자장 하기 위해
from langchain.vectorstores import FAISS # vectorstore 저장하기 위해



def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)

    return len(tokens)


# 업로드한 파일을 모두 text로 전환하는 함수
def get_text(docs):
    doc_list = []  # 여러 개의 파일 처리를 위해 선언

    for doc in docs:
        file_name = doc.name  # doc 객체의 이름을 파일 이름으로 사용, 불러온 파일이름을 file_name에 저장
        with open(file_name, "wb") as file:  # 파일을 doc.name으로 저장, file_name을 열고
            file.write(doc.getvalue())  # 원래 doc에 있는 value들을 이 file에 적는 다는 것
            logger.info(f"Uploaded {file_name}")  # file_name 업로드 한 내역을 업로드 해서 로그를 남김

        # 클라우드 상에서 다양한 파일을 처리해주기 위해
        if '.pdf' in doc.name:  # pdf 문서라면
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()  # 페이지별로 분할
        elif '.docx' in doc.name:
            loader = Docx2txtLoader(file_name)
            documents = loader.load_and_split()
        elif '.pptx' in doc.name:
            loader = UnstructuredPowerPointLoader(file_name)
            documents = loader.load_and_split()

        doc_list.extend(documents)  # extend를 통해서 documents 값들을 doc_list에 저장

    return doc_list


# chunk split
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks


# chunk들을 벡터화
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(  # 임베딩 모델 선언
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}  # 벡터저장소에 저장해서 사용자의 질문과 비교하기 위해
    )
    vectordb = FAISS.from_documents(text_chunks, embeddings)  # 벡터저장소 선언

    return vectordb


# 위에서 선언한 함수들 구현을 위한 함수
def get_conversation_chain(vetorestore, openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-3.5-turbo', temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vetorestore.as_retriever(search_type='mmr', vervose=True),
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
        # 메모리를 사용하기 위해, chat_history라는 key값을 가진 chat기록을 찾아서 context에 집어넣어서 llm이 답변할 때 이전 대답을 찾아보도록 함
        # output_key='answer' : 대화를 나눈 것 중 answer에 해당하는 부분만 저장
        get_chat_history=lambda h: h,  # lambda h: h = 메모리가 들어온 그대로 get_chat_history에 저장
        return_source_documents=True,  # llm이 참고한 문서를 반환
        verbose=True
    )

    return conversation_chain