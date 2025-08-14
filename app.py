import streamlit as st
import os
import time
import uuid
from rag_logic import (
    prepare_vectorstore,        # 벡터스토어 준비 (저장/로드)
    build_retriever,            # 검색기 생성
    build_streaming_llm,        # 스트리밍 LLM 생성
    make_context_and_sources,   # 질문→컨텍스트/출처/서식 추출
    build_final_prompt          # 프롬프트 생성
)


# === 1. 세션 상태 초기화 ===
def init_session_state():
    defaults = {
        "messages": [],
        "api_key": None,
        "question_count": 0,
        "processing": False,
        "selected_question": None,
        "last_clicked_question": None,
        "vector_dir": "faiss_minweonpyeonram_2025",
        "pdf_path": "minweonpyeonram-2025.pdf",
        "index_ready": False,
        "retriever": None,
        "file_names": ["곡성군 민원편람 2025"]
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# === 2. 메인 ===
def main():
    init_session_state()
    st.set_page_config(
        page_title="🏛️ 곡성군 AI 민원상담봇",
        page_icon="🏛️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 다크 모드 & 모바일 최적화
    st.markdown("""
    <style>
        :root { color-scheme: dark; }
        body, .stApp { background-color: #121212; color: #fff; }
        .main-header {
            background: linear-gradient(90deg, #222 0%, #444 100%);
            padding: 1.2rem; border-radius: 10px;
            text-align: center; margin-bottom: 1rem;
        }
        .main-header h2 { margin: 0; color: #fff; }
        .main-header p { margin: 0; font-size: 0.9em; color: #bbb; }
        .metric-card {
            background: #1e1e1e; color: #eee;
            padding: 1rem; border-radius: 8px;
            border-left: 4px solid #667eea; margin-bottom: 1rem;
            font-size: 0.95em;
        }
        .footer {
            padding: 0.8rem; text-align: center;
            font-size: 0.8em; color: #aaa; border-top: 1px solid #333;
            margin-top: 1.5rem;
        }
        @media (max-width: 768px) {
            .main-header h2 { font-size: 1.2em; }
            .main-header p { font-size: 0.8em; }
        }
    </style>
    """, unsafe_allow_html=True)

    # 헤더
    st.markdown("""
    <div class="main-header">
        <h2>🏛️ 곡성군 AI 민원상담봇</h2>
        <p>곡성군 민원편람 기반 AI 상담 서비스</p>
    </div>
    """, unsafe_allow_html=True)

    setup_sidebar()

    if not st.session_state.api_key:
        st.warning("🔑 사이드바에서 OpenAI API 키를 입력해주세요.")
        st.stop()

    initialize_system()
    display_chat_interface()
    display_footer()


# === 3. 사이드바 ===
def setup_sidebar():
    st.sidebar.title("API 설정")
    key = st.sidebar.text_input("OpenAI API 키", type="password", key="api_key_input")
    if key:
        st.session_state.api_key = key

    st.sidebar.markdown("---")
    st.sidebar.subheader("빠른 질문")
    quick_qs = [
        "여권을 발급 받고 싶어요",
        "전입신고 방법을 알고 싶어요요",
        "인감증명서 발급 받고 싶어요",
        "정보공개를 청구방법을 알고 싶어요",
        "건축허가 신청 절차를 알고 싶어요"
    ]
    for q in quick_qs:
        if st.sidebar.button(q, key=f"btn_{q}"):
            if not st.session_state.processing and st.session_state.last_clicked_question != q:
                st.session_state.selected_question = q
                st.session_state.last_clicked_question = q

    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ 대화 초기화"):
        st.session_state.messages.clear()
        st.session_state.question_count = 0
        st.session_state.selected_question = None
        st.session_state.last_clicked_question = None
        st.experimental_rerun()


# === 4. 시스템 초기화 ===
def initialize_system():
    pdf_path = st.session_state.pdf_path
    vector_dir = st.session_state.vector_dir

    if not os.path.exists(pdf_path):
        st.error(f"❌ '{pdf_path}' 파일이 없습니다.")
        st.stop()

    if not st.session_state.index_ready:
        with st.spinner("📄 인덱스 준비 중..."):
            vectorstore = prepare_vectorstore(
                openai_api_key=st.session_state.api_key,
                pdf_paths=[pdf_path],
                file_names=st.session_state.file_names,
                vector_dir=vector_dir
            )
            st.session_state.retriever = build_retriever(vectorstore, k=8)
            st.session_state.index_ready = True


# === 5. 채팅 인터페이스 ===
def display_chat_interface():
    st.markdown(
        f"<div class='metric-card'>📄 문서: <b>{', '.join(st.session_state.file_names)}</b> | 💬 질문 수: {st.session_state.question_count}</div>",
        unsafe_allow_html=True
    )

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if st.session_state.selected_question and not st.session_state.processing:
        q = st.session_state.selected_question
        st.session_state.selected_question = None
        process_question_typing(q)

    if not st.session_state.processing:
        if prompt := st.chat_input("✍️ 민원업무 질문을 입력하세요..."):
            process_question_typing(prompt)


# === 6. 질문 처리 (타자기 스타일) ===
def process_question_typing(prompt, delay=0.02):
    if st.session_state.processing:
        return
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user" and \
       st.session_state.messages[-1]["content"] == prompt:
        return

    st.session_state.processing = True
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.question_count += 1

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            from_html = st.empty()
            with st.spinner("🤖 답변 생성 중..."):
                # 컨텍스트/서식 추출
                context_text, _, annex_forms = make_context_and_sources(
                    st.session_state.retriever, prompt
                )

                # LLM
                llm = build_streaming_llm(
                    model="gpt-4o-mini",
                    openai_api_key=st.session_state.api_key,
                    max_tokens=800,
                    temperature=0
                )

                final_prompt = build_final_prompt(
                    context=context_text,
                    question=prompt,
                    annex_forms=annex_forms
                )

                full_text = ""
                for chunk in llm.stream(final_prompt):
                    token = getattr(chunk, "content", None) or str(chunk)
                    full_text += token
                    from_html.markdown(full_text)
                    time.sleep(delay)

                # 메타데이터, 처리시간 제거 -> 그대로 저장
                st.session_state.messages.append({"role": "assistant", "content": full_text})

        except Exception as e:
            err_msg = f"❌ 오류: {e}"
            st.error(err_msg)
            st.session_state.messages.append({"role": "assistant", "content": err_msg})

    st.session_state.processing = False


# === 7. 푸터 ===
def display_footer():
    st.markdown("""
    <div class="footer">
        🏛 곡성군청 | 📞 061-360-0000 | 🌐 www.gokseong.go.kr | 📍 전남 곡성군 곡성읍 군청로 15  
        ⚠ 본 서비스는 AI 안내 서비스이며, 정확한 민원은 담당부서에 문의하세요.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()



