# app.py
import streamlit as st
import os
import time
import uuid
from rag_logic import initialize_rag_chain, get_answer


# ---- [1] 세션 상태 초기화 ----
def init_session_state():
    defaults = {
        "messages": [],
        "rag_chain": None,
        "retriever": None,
        "api_key": None,
        "file_hash": None,
        "file_names": [],
        "chat_id": str(uuid.uuid4()),
        "user_feedback": {},
        "question_count": 0,
        "processing": False,
        "selected_question": None,
        "last_clicked_question": None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ---- [2] 메인 실행 함수 ----
def main():
    init_session_state()

    st.set_page_config(
        page_title="🏛️ 곡성군 AI 민원상담봇",
        page_icon="🏛️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # ---- 스타일 ----
    st.markdown("""
    <style>
        body { font-family: 'Noto Sans KR', sans-serif; }
        .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 1.5rem;
        }
        .metric-card {
            background: #f9f9f9;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            margin-bottom: 1rem;
        }
        .chat-input textarea {
            border: 2px solid #667eea !important;
            border-radius: 6px !important;
            font-size: 16px !important;
        }
        .footer {
            padding: 0.8rem;
            text-align: center;
            font-size: 0.85em;
            color: #666;
            border-top: 1px solid #ddd;
            margin-top: 2rem;
        }
    </style>
    """, unsafe_allow_html=True)

    # ---- 헤더 ----
    st.markdown("""
    <div class="main-header">
        <h2>🏛️ 곡성군 AI 민원상담봇</h2>
        <p>곡성군 민원편람 기반 AI 상담 서비스</p>
    </div>
    """, unsafe_allow_html=True)

    # ---- 사이드바 ----
    setup_sidebar()

    # API 키 체크
    if not st.session_state.api_key:
        st.warning("🔑 사이드바에서 OpenAI API 키를 입력해주세요.")
        st.stop()

    # 초기화
    initialize_system()
    display_chat_interface()
    display_footer()


# ---- [3] 사이드바 ----
def setup_sidebar():
    st.sidebar.title("🔑 API 설정")
    api_key = st.sidebar.text_input(
        "OpenAI API 키 입력",
        type="password",
        placeholder="sk-...",
        key="api_key_input"
    )
    if api_key:
        st.session_state.api_key = api_key

    st.sidebar.markdown("---")
    st.sidebar.subheader("🚀 빠른 질문")
    quick_qs = [
        "여권을 발급 받고 싶어요",
        "정보공개 청구 시 필요한 서류는?",
        "인감증명서 발급에 필요한 서류는?",
        "주민등록등본 발급에 필요한 서류는?",
        "건축허가 신청 시 필요한 서류는?"
    ]
    for q in quick_qs:
        if st.sidebar.button(q, key=f"btn_{q}"):
            if not st.session_state.processing and st.session_state.last_clicked_question != q:
                st.session_state.selected_question = q
                st.session_state.last_clicked_question = q

    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ 대화 초기화"):
        st.session_state.clear()
        init_session_state()
        st.experimental_rerun()


# ---- [4] 시스템 초기화 ----
def initialize_system():
    pdf_path = "minweonpyeonram-2025.pdf"
    if not os.path.exists(pdf_path):
        st.error("❌ 'minweonpyeonram-2025.pdf' 파일이 없습니다.")
        st.stop()

    file_hash = str(hash(open(pdf_path, 'rb').read()))
    if not st.session_state.rag_chain or st.session_state.file_hash != file_hash:
        with st.spinner("📄 문서를 분석 중입니다..."):
            rag_chain, retriever, _ = initialize_rag_chain(
                st.session_state.api_key, [pdf_path], ["곡성군 민원편람 2025"]
            )
            st.session_state.rag_chain = rag_chain
            st.session_state.retriever = retriever
            st.session_state.file_hash = file_hash
            st.session_state.file_names = ["곡성군 민원편람 2025"]


# ---- [5] 채팅 UI ----
def display_chat_interface():
    st.markdown(f"""
    <div class="metric-card">
        📄 현재 문서: <b>{', '.join(st.session_state.file_names)}</b>  
        💬 총 질문 수: {st.session_state.question_count}
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📖 사용 안내", expanded=False):
        st.markdown("""
        - **사이드바**에서 빠른 질문 클릭  
        - **채팅창**에 질문 입력  
        - 가능한 질문 예시:  
          • "여권을 발급 받고 싶어요"  
          • "정보공개 청구 처리기간은?"  
          • "건축허가 신청 서류는?"
        """)

    # 이전 대화 표시
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 빠른 질문 처리
    if st.session_state.selected_question and not st.session_state.processing:
        question = st.session_state.selected_question
        st.session_state.selected_question = None
        process_question(question)

    # 입력창
    if not st.session_state.processing:
        if prompt := st.chat_input("✍️ 민원업무 질문을 입력하세요..."):
            process_question(prompt)


# ---- [6] 질문 처리 ----
def process_question(prompt):
    # 중복 방지
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
        with st.spinner("🤖 답변 생성 중..."):
            try:
                start = time.time()
                answer = get_answer(
                    st.session_state.rag_chain,
                    st.session_state.retriever,
                    prompt,
                    st.session_state.api_key
                )
                elapsed = round(time.time() - start, 2)
                full_ans = f"{answer}\n\n_⏱️ 응답 시간: {elapsed}초_"
                st.markdown(full_ans)
                st.session_state.messages.append({"role": "assistant", "content": full_ans})
            except Exception as e:
                err = f"❌ 오류 발생: {e}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})

    st.session_state.processing = False


# ---- [7] 푸터 ----
def display_footer():
    st.markdown("""
    <div class="footer">
        🏛️ 곡성군청 | 📞 061-360-0000 | 🌐 www.gokseong.go.kr | 📍 전남 곡성군 곡성읍 군청로 15  
        ⚠️ 본 서비스는 AI 기반 안내 서비스로, 정확한 민원은 담당부서에 문의 바랍니다.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()






