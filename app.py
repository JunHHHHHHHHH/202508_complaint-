# app.py
import streamlit as st
import os
import time
import uuid
from rag_logic import initialize_rag_chain
from langchain_openai import ChatOpenAI


# ===== 1. 세션 상태 초기화 =====
def init_session_state():
    defaults = {
        "messages": [],
        "rag_chain": None,
        "retriever": None,
        "api_key": None,
        "file_hash": None,
        "file_names": [],
        "chat_id": str(uuid.uuid4()),
        "question_count": 0,
        "processing": False,
        "selected_question": None,
        "last_clicked_question": None
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ===== 2. 메인 =====
def main():
    init_session_state()
    st.set_page_config(
        page_title="🏛️ 곡성군 AI 민원상담봇",
        page_icon="🏛️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 다크 모드 + 모바일 스타일
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
            .metric-card { font-size: 0.85em; padding: 0.8rem; }
            .footer { font-size: 0.7em; padding: 0.5rem; }
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


# ===== 3. 사이드바 =====
def setup_sidebar():
    st.sidebar.title("API 설정")
    key = st.sidebar.text_input("OpenAI API 키", type="password", key="api_key_input")
    if key:
        st.session_state.api_key = key

    st.sidebar.markdown("---")
    st.sidebar.subheader("빠른 질문")
    quick_qs = [
        "여권을 발급 받고 싶어요",
        "주민등록등본을 발급 받고 싶어요",
        "인감증명서를 발급 받고 싶어요",
        "정보공개를 청구하고 싶어요",
        "건축허가 신청을 하고 싶어요"
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


# ===== 4. 시스템 초기화 =====
def initialize_system():
    pdf_path = "minweonpyeonram-2025.pdf"
    if not os.path.exists(pdf_path):
        st.error("❌ 'minweonpyeonram-2025.pdf' 파일이 없습니다.")
        st.stop()

    file_hash = str(hash(open(pdf_path, "rb").read()))
    if not st.session_state.rag_chain or st.session_state.file_hash != file_hash:
        with st.spinner("📄 곡성군 민원편람(2025) 문서 분석 중..."):
            rag_chain, retriever, _ = initialize_rag_chain(
                st.session_state.api_key, [pdf_path], ["곡성군 민원편람 2025"]
            )
            st.session_state.rag_chain = rag_chain
            st.session_state.retriever = retriever
            st.session_state.file_hash = file_hash
            st.session_state.file_names = ["곡성군 민원편람 2025"]


# ===== 5. 채팅 인터페이스 =====
def display_chat_interface():
    st.markdown(
        f"<div class='metric-card'>📄 문서: <b>{', '.join(st.session_state.file_names)}</b> | 💬 질문 수: {st.session_state.question_count}</div>",
        unsafe_allow_html=True
    )

    with st.expander("사용 안내", expanded=False):
        st.markdown("""
        • 사이드바에서 빠른 질문 클릭  
        • 하단 채팅창에 직접 입력  
        • 예시: "여권을 발급 받고 싶어요", "부동산 거래 시 신고방법을 알고 싶어요"
        """)

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


# ===== 6. 타자기 스타일 순차 출력 =====
def process_question_typing(prompt, delay=0.02):
    """LLM 답변을 한 글자씩 순차적으로 출력"""
    if st.session_state.processing:
        return
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user" \
       and st.session_state.messages[-1]["content"] == prompt:
        return

    st.session_state.processing = True
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.question_count += 1

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("🤖 답변 생성 중..."):
                start_time = time.time()

                # 문서 검색
                docs = st.session_state.retriever.get_relevant_documents(prompt)
                context = "\n\n".join(
                    [f"[출처: {d.metadata.get('source_info','?')}] {d.page_content}" for d in docs]
                )

                # LLM 준비 (스트리밍)
                llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0,
                    openai_api_key=st.session_state.api_key,
                    max_tokens=800,
                    streaming=True
                )

                prompt_text = f"""
당신은 곡성군의 민원 상담 전문가입니다.
곡성군 민원편람을 기반으로 정확하고 친절하게 그리고 자세히 답변하세요.
관련된 별지 서식도 함께 알려주세요. 

문맥:
{context}

질문:
{prompt}

답변:
"""

                container = st.empty()
                full_text = ""
                for chunk in llm.stream(prompt_text):
                    token = chunk.content if hasattr(chunk, "content") else str(chunk)
                    full_text += token
                    container.markdown(full_text)
                    time.sleep(delay)  # 타자 속도 조절

                elapsed = round(time.time() - start_time, 2)
                full_text += f"\n\n_⏱ {elapsed}초_"
                container.markdown(full_text)
                st.session_state.messages.append({"role": "assistant", "content": full_text})

        except Exception as e:
            err = f"❌ 오류: {e}"
            st.error(err)
            st.session_state.messages.append({"role": "assistant", "content": err})

    st.session_state.processing = False


# ===== 7. 푸터 =====
def display_footer():
    st.markdown("""
    <div class="footer">
        🏛 곡성군청 | 📞 061-360-0000 | 🌐 www.gokseong.go.kr | 📍 전남 곡성군 곡성읍 군청로 15  
        ⚠ 본 서비스는 AI 안내 서비스이며, 정확한 민원은 담당부서에 문의하세요.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()



