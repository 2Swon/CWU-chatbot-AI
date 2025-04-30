import streamlit as st
import requests
import os
import json
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

# 환경 변수 로드
load_dotenv()

# API URL
API_URL = "http://localhost:8000"

# 페이지 설정
st.set_page_config(
    page_title="청운대학교 챗봇",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Streamlit 앱 타이틀 및 설명
st.title("청운대학교 챗봇 AI 시스템")
st.markdown("""
이 챗봇은 청운대학교 관련 정보를 제공합니다. 
두 가지 AI 모델(LangChain과 GraphRAG)을 비교해볼 수 있습니다.
""")

# 사이드바 설정
st.sidebar.title("설정")
model_option = st.sidebar.radio(
    "AI 모델 선택",
    ["LangChain", "GraphRAG", "비교 모드"]
)

# 함수: API 엔드포인트 호출
def get_ai_response(question, mode="langchain"):
    try:
        if mode == "compare":
            response = requests.post(
                f"{API_URL}/compare",
                json={"question": question}
            )
        else:
            response = requests.post(
                f"{API_URL}/question",
                json={"question": question, "mode": mode}
            )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API 호출 오류: {response.status_code} - {response.text}"}
    except Exception as e:
        return {"error": f"요청 실패: {str(e)}"}

# 함수: 평가 실행
def run_evaluation(sample_count=10):
    try:
        response = requests.post(
            f"{API_URL}/evaluate",
            json={"sample_count": sample_count}
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API 호출 오류: {response.status_code} - {response.text}"}
    except Exception as e:
        return {"error": f"요청 실패: {str(e)}"}

# 채팅 기록을 세션 상태로 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 기본 탭 설정
tab1, tab2 = st.tabs(["챗봇 인터페이스", "성능 평가"])

# 탭 1: 챗봇 인터페이스
with tab1:
    # 채팅 기록 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "model" in message:
                st.caption(f"모델: {message['model']}")
    
    # 사용자 입력 처리
    if prompt := st.chat_input("청운대학교에 관해 질문해보세요..."):
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # 사용자 메시지 표시
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # AI 응답 생성
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("🤔 생각 중...")
            
            if model_option == "LangChain":
                # LangChain 모델 사용
                response = get_ai_response(prompt, "langchain")
                if "error" in response:
                    full_response = f"오류가 발생했습니다: {response['error']}"
                else:
                    full_response = response["answer"]
                
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response, "model": "LangChain"})
                
            elif model_option == "GraphRAG":
                # GraphRAG 모델 사용
                response = get_ai_response(prompt, "graphrag")
                if "error" in response:
                    full_response = f"오류가 발생했습니다: {response['error']}"
                else:
                    full_response = response["answer"]
                
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response, "model": "GraphRAG"})
                
            else:  # 비교 모드
                response = get_ai_response(prompt, "compare")
                if "error" in response:
                    full_response = f"오류가 발생했습니다: {response['error']}"
                    message_placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response, "model": "비교 모드"})
                else:
                    # 두 개의 응답을 나란히 표시
                    message_placeholder.empty()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("LangChain 응답:")
                        st.markdown(response["langchain_answer"])
                    
                    with col2:
                        st.subheader("GraphRAG 응답:")
                        st.markdown(response["graphrag_answer"])
                    
                    # 결합된 응답을 메시지 기록에 저장
                    combined_response = f"**LangChain 응답:**\n{response['langchain_answer']}\n\n**GraphRAG 응답:**\n{response['graphrag_answer']}"
                    st.session_state.messages.append({"role": "assistant", "content": combined_response, "model": "비교 모드"})

# 탭 2: 성능 평가
with tab2:
    st.header("모델 성능 평가")
    st.markdown("""
    이 섹션에서는 LangChain과 GraphRAG 접근 방식의 성능을 RAGAS 메트릭을 사용하여 비교합니다.
    테스트 질문 수를 선택하고 평가를 실행하세요.
    """)
    
    # 평가 설정
    col1, col2 = st.columns([3, 1])
    with col1:
        sample_count = st.slider("테스트 질문 수", min_value=5, max_value=20, value=10, step=1)
    
    with col2:
        eval_button = st.button("평가 실행", type="primary")
    
    # 평가 실행
    if eval_button:
        with st.spinner("평가 실행 중... (몇 분 정도 소요될 수 있습니다)"):
            results = run_evaluation(sample_count)
            
            if "error" in results:
                st.error(f"평가 중 오류 발생: {results['error']}")
            else:
                st.success("평가 완료!")
                
                # 결과 처리
                try:
                    evaluation_data = results["results"]
                    
                    # 차트 데이터 준비
                    metrics = []
                    langchain_scores = []
                    graphrag_scores = []
                    diffs = []
                    
                    for metric, score in evaluation_data["LangChain"].items():
                        metrics.append(metric)
                        langchain_scores.append(score)
                        graphrag_scores.append(evaluation_data["GraphRAG"][metric])
                        diffs.append(evaluation_data["diff"][metric])
                    
                    # 결과 시각화
                    st.subheader("성능 메트릭 비교")
                    
                    # 데이터프레임 생성
                    df = pd.DataFrame({
                        'Metric': metrics * 2,
                        'System': ["LangChain"] * len(metrics) + ["GraphRAG"] * len(metrics),
                        'Score': langchain_scores + graphrag_scores
                    })
                    
                    # 차트 생성
                    fig, ax = plt.subplots(figsize=(10, 6))
                    colors = ["#1f77b4", "#2ca02c"]
                    
                    bars = sns.barplot(x='Metric', y='Score', hue='System', data=df, ax=ax, palette=colors)
                    
                    ax.set_title('LangChain vs GraphRAG 성능 비교', fontsize=16)
                    ax.set_xlabel('메트릭', fontsize=14)
                    ax.set_ylabel('점수', fontsize=14)
                    ax.set_ylim(0, 1.0)
                    
                    # 그리드 추가
                    ax.grid(axis='y', linestyle='--', alpha=0.7)
                    
                    # 레전드 위치 조정
                    ax.legend(title='시스템', bbox_to_anchor=(1.05, 1), loc='upper left')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # 상세 결과 테이블
                    st.subheader("상세 결과")
                    
                    result_df = pd.DataFrame({
                        '메트릭': metrics,
                        'LangChain': langchain_scores,
                        'GraphRAG': graphrag_scores,
                        '차이 (GraphRAG - LangChain)': diffs
                    })
                    
                    # 차이에 따라 색상 하이라이팅
                    def highlight_diff(val):
                        if isinstance(val, float):
                            if val > 0:
                                return 'background-color: rgba(0, 128, 0, 0.2)'  # 녹색 (양수)
                            elif val < 0:
                                return 'background-color: rgba(255, 0, 0, 0.2)'  # 빨간색 (음수)
                        return ''
                    
                    styled_df = result_df.style.applymap(highlight_diff, subset=['차이 (GraphRAG - LangChain)'])
                    styled_df = styled_df.format({
                        'LangChain': '{:.3f}',
                        'GraphRAG': '{:.3f}',
                        '차이 (GraphRAG - LangChain)': '{:.3f}'
                    })
                    
                    st.dataframe(styled_df, use_container_width=True)
                    
                    # 결론
                    st.subheader("결론")
                    
                    avg_diff = sum(diffs) / len(diffs)
                    if avg_diff > 0:
                        st.success(f"GraphRAG가 평균 {avg_diff:.3f} 점 더 높은 성능을 보입니다.")
                    elif avg_diff < 0:
                        st.error(f"LangChain이 평균 {-avg_diff:.3f} 점 더 높은 성능을 보입니다.")
                    else:
                        st.info("두 접근 방식의 평균 성능이 동일합니다.")
                    
                    # 메트릭별 분석
                    for i, metric in enumerate(metrics):
                        if diffs[i] > 0.05:
                            st.write(f"📈 **{metric}**: GraphRAG가 {diffs[i]:.3f} 점 더 높습니다.")
                        elif diffs[i] < -0.05:
                            st.write(f"📉 **{metric}**: LangChain이 {-diffs[i]:.3f} 점 더 높습니다.")
                        else:
                            st.write(f"📊 **{metric}**: 두 접근 방식의 성능이 비슷합니다 (차이: {diffs[i]:.3f}).")
                except Exception as e:
                    st.error(f"평가 결과 처리 오류: {str(e)}")
                    st.error("결과 정보 수집을 실패했습니다. 로그를 확인해보세요.")
                    st.json(results)  # 원래 결과를 JSON으로 표시

# 푸터 추가
st.markdown("---")
st.markdown("© 2025 청운대학교 챗봇 AI 시스템")
