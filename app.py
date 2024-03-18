import streamlit as st
from utils import get_text, get_text_chunks, get_vectorstore, get_conversation_chain
# from streamlit_chat import message
from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory

def run_app():
    st.set_page_config(  # streamlit 페이지의 상세 설정
        page_title="DirChat",  # 페이지 이름
        page_icon=":books:")  # 페이지 아이콘

    st.title("_Private Data :red[QA Chat]_ :books:")
    # 페이지 내 제목
    # __ : 글자 눕히기
    # :books: : 책 아이콘

    if "conversation" not in st.session_state:
        st.session_state.conversation = None  # 미리 none이라고 선언해야 코드 뒤 부분에서 오류 방지

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None  # 미리 none이라고 선언해야 코드 뒤 부분에서 오류 방지

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None  # 미리 none이라고 선언해야 코드 뒤 부분에서 오류 방지

    # sidebar
    with st.sidebar:

        uploaded_files = st.file_uploader("Upload your file", type=['pdf', 'docx'],
                                          accept_multiple_files=True)  # file_uploader 기능
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")  # text_input
        process = st.button("Process")  # Process 버튼

    if process:  # Process 버튼을 누른 경우

        if not openai_api_key:  # 가장 먼저 openai_api_key 입력 여부 확인
            st.info("Please add your OpenAI API key to continue.")
            st.stop()

        # 만약 openai_api_key가 입력 되었다면
        files_text = get_text(uploaded_files)  # 파일 load
        text_chunks = get_text_chunks(files_text)  # text를 chunk로 분할
        vetorestore = get_vectorstore(text_chunks)  # 벡터화

        st.session_state.conversation = get_conversation_chain(vetorestore, openai_api_key)
        # get_conversation_chain : vetorestore을 통해 llm이 답변할 수 있도록 chain을 구성

        # process버튼을 누르면 업로드된 파일을 text -> 벡터화 -> api_key 확인 -> llm

        st.session_state.processComplete = True

    # 채팅화면 구현하기
    if 'messages' not in st.session_state:  # assistant 메시지 초기값으로 아래 문장을 넣어줌
        st.session_state['messages'] = [{"role": "assistant",
                                         "content": "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):  # message 마다 with 문으로 묶어 주는 것
            st.markdown(message["content"])  # 어떤 역할에 따라 아이콘을 부여하고 content에 해당하는 문장을 적어라

    history = StreamlitChatMessageHistory(key="chat_messages")  # 이전 답변을 참고하기 위해 history 생성

    # Chat logic
    # 질의응답
    if query := st.chat_input("질문을 입력해주세요."):  # 사용자가 질문을 입력 한다면
        st.session_state.messages.append({"role": "user", "content": query})  # 먼저 user 역할을 부여하고 content에 질문을 넣어 준다

        with st.chat_message("user"):  # user
            st.markdown(query)  # user 아이콘

        # assistant가 답변
        with st.chat_message("assistant"):
            chain = st.session_state.conversation

            with st.spinner("Thinking..."):  # 로딩 중 이라는 것을 기호로 시각화 및 Thinking...을 보여줌
                result = chain(
                    {"question": query})  # chain = session_sate.conversation(), query를 llm에 넣어주면 나오는 답변을 result에 저장

                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']
                    # 이전에 주고 받은 답변 기록을 chat_history에 저장
                response = result['answer']  # answer를 response에 저장
                source_documents = result.get('source_documents', [])  # 참고한 문서를 source_documents에 저장, 기본값으로 빈 리스트 설정

                st.markdown(response)  # assistant 아이콘 옆 적히는 컨텐츠

                # Check if source_documents is not empty before trying to access it
                if source_documents:
                    with st.expander("참고 문서 확인"):
                        # Ensure that you access only indexes that exist within the list
                        for i in range(
                                min(3, len(source_documents))):  # Show up to 3 documents, but not more than available
                            st.markdown(source_documents[i].metadata['source'], help=source_documents[i].page_content)
                else:
                    st.write("No source documents are available for display.")

        # Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        # session_state에 assistant가 답변한 것도 저장함