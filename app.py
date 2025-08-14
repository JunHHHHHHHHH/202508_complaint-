# app.py

import streamlit as st
import uuid
import os

from rag_logic import initialize_rag_chain, get_answer

APP_TITLE = "🏛️ 곡성군 AI 민원상담봇"
PDF_PATH = "./minweonpyeonram-2025.pdf"
DISPLAY_NAME = "minweonpyeonram-2025.pdf"

def init_session():
    defaults = {
        "messages": [],
        "rag_chain": None,
        "retriever": None,
        "api_key": None,
        "chat_id": str(uuid.uuid4()),
        "question_count": 0,
        "processing": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def build_header():
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🏛️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title(APP_TITLE)
    st.caption("곡성군 민원편람 기반 AI 상담 서비스 (어르신도 쉽게 이해할 수 있도록 안내합니다)")

def sidebar():
    with st.sidebar:
        st.subheader("환경 설정")
        api_key = st.text_input("OpenAI API Key", type="password", value=st.session_state.get("api_key") or os.getenv("OPENAI_API_KEY", ""))
        if api_key:
            st.session_state["api_key"] = api_key

        st.markdown("---")
        st.subheader("PDF 정보")
        st.write(f"문서: {DISPLAY_NAME}")
        exists = os.path.exists(PDF_PATH)
        st.write("파일 상태: " + ("✅ 발견됨" if exists else "❌ 없음"))

        if st.button("RAG 초기화/갱신"):
            if not st.session_state["api_key"]:
                st.error("API Key를 입력해주세요.")
            elif not exists:
                st.error("PDF 파일을 찾을 수 없습니다.")
            else:
                with st.spinner("RAG 체인 초기화 중... (필요 시 임베딩/인덱스 생성)"):
                    chain, retriever = initialize_rag_chain(st.session_state["api_key"], PDF_PATH, DISPLAY_NAME)
                    st.session_state["rag_chain"] = chain
                    st.session_state["retriever"] = retriever
                st.success("초기화 완료")

def main_area():
    st.markdown("아래 입력창에 궁금한 민원 내용을 쓰면, 어르신도 이해하기 쉽게 안내해 드립니다.")
    question = st.text_input("질문 입력 (예: 여권 재발급하려면 무엇이 필요합니까?)")

    col1, col2 = st.columns([1, 1])
    with col1:
        ask = st.button("질문하기")
    with col2:
        clear = st.button("대화 초기화")

    if clear:
        st.session_state["messages"] = []
        st.session_state["question_count"] = 0
        st.success("대화가 초기화되었습니다.")

    if ask:
        if not st.session_state["api_key"]:
            st.error("API Key를 먼저 입력해주세요.")
            return
        if not st.session_state.get("rag_chain") or not st.session_state.get("retriever"):
            try:
                with st.spinner("검색 엔진 준비 중..."):
                    chain, retriever = initialize_rag_chain(st.session_state["api_key"], PDF_PATH, DISPLAY_NAME)
                    st.session_state["rag_chain"] = chain
                    st.session_state["retriever"] = retriever
            except Exception as e:
                st.error(f"초기화 실패: {e}")
                return

        with st.spinner("답변 생성 중..."):
            try:
                answer = get_answer(st.session_state["rag_chain"], st.session_state["retriever"], question, st.session_state["api_key"])
            except Exception as e:
                st.error(f"오류: {e}")
                return

        st.session_state["messages"].append({"role": "user", "content": question})
        st.session_state["messages"].append({"role": "assistant", "content": answer})
        st.session_state["question_count"] += 1

    # 대화 표시
    for m in st.session_state["messages"]:
        if m["role"] == "user":
            st.markdown(f"👤 질문\n\n{m['content']}")
        else:
            st.markdown(f"🤖 답변\n\n{m['content']}")

def main():
    init_session()
    build_header()
    sidebar()
    main_area()

if __name__ == "__main__":
    main()


