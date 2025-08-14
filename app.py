# app.py
import streamlit as st
import os
import time
import re
from rag_logic import (
    prepare_vectorstore,
    build_retriever,
    build_streaming_llm,
    make_context_and_sources,
    build_final_prompt
)

# ---------------------------
# CSS 테마 (Buybrand 스타일 톤 앤 매너)
# ---------------------------
THEME_CSS = """
<style>
:root {
  --brand-bg: #0b0b0c;
  --card-bg: #111214;
  --muted: #8B8D93;
  --text: #F5F6F8;
  --accent: #A6E3A1;
  --accent-2: #89B4FA;
  --border: #23252A;
}
.stApp {
  background: var(--brand-bg);
  color: var(--text);
  font-family: "Noto Sans KR", sans-serif;
}
h1,h2,h3 { color: var(--text); letter-spacing: -0.02em; }
.hero {
  padding: 28px 24px;
  border: 1px solid var(--border);
  border-radius: 16px;
  background: linear-gradient(180deg,#121316 0%,#101114 100%);
  margin-bottom: 18px;
}
.hero-eyebrow { color: var(--muted); font-size: 14px; margin-bottom: 6px;}
.hero-title { font-size: 26px; font-weight: 800; margin: 6px 0 8px; }
.hero-desc { color: var(--muted); font-size: 15px; }
.card {
  border: 1px solid var(--border);
  border-radius: 14px;
  background: var(--card-bg);
  padding: 18px;
  margin-bottom: 14px;
}
textarea, input, .stTextInput>div>div>input {
  background: #0D0E10 !important;
  color: var(--text) !important;
  border: 1px solid #24262B !important;
  border-radius: 10px !important;
}
.stButton>button {
  background: var(--accent-2);
  color: #0b0b0c;
  border-radius: 10px;
  font-weight: 700;
}
.stButton>button:hover { filter: brightness(0.95); }
.msg-bot {
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 14px;
  background: #0E0F12;
}
.pill {
  display: inline-block;
  padding: 6px 10px;
  border: 1px solid var(--border);
  border-radius: 999px;
  font-size: 12px;
  color: var(--muted);
  margin-right: 6px;
}
.foot { color: var(--muted); font-size: 12px; text-align: right; margin-top: 6px; }
</style>
"""

# ---------------------------
# 세션 초기화
# ---------------------------
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
        "file_names": ["곡성군 민원편람 2025"],
        "typing_delay": 0.02,  # 타자 속도
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ---------------------------
# Hero 영역
# ---------------------------
def render_hero():
    st.markdown(THEME_CSS, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="hero">
      <div class="hero-eyebrow">🏛️ 곡성군 AI 민원상담봇</div>
      <div class="hero-title">민원, 더 간결하고 빠르게 해결해요.</div>
      <div class="hero-desc">곡성군 민원편람 기반으로 답변 드립니다.</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------
# 앱 메인
# ---------------------------
def main():
    init_session_state()
    st.set_page_config(
        page_title="곡성군 AI 민원상담봇",
        page_icon="🏛️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    render_hero()
    setup_sidebar()

    if not st.session_state.api_key:
        st.warning("🔑 사이드바에서 OpenAI API 키를 입력해주세요.")
        st.stop()

    initialize_system()
    display_chat_interface()
    display_footer()

# ---------------------------
# 사이드바 설정
# ---------------------------
def setup_sidebar():
    st.sidebar.title("API 설정")
    key = st.sidebar.text_input("OpenAI API 키", type="password", key="api_key_input")
    if key:
        st.session_state.api_key = key

    st.sidebar.markdown("---")
    st.sidebar.subheader("빠른 질문")
    quick_qs = [
        "여권을 발급 받고 싶어요",
        "전입신고 방법을 알고 싶어요",
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

# ---------------------------
# 시스템 초기화
# ---------------------------
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

# ---------------------------
# 채팅 UI
# ---------------------------
def display_chat_interface():
    st.markdown(
        f"<div class='card'>📄 문서: <b>{', '.join(st.session_state.file_names)}</b> | 💬 질문 수: {st.session_state.question_count}</div>",
        unsafe_allow_html=True
    )

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if st.session_state.selected_question and not st.session_state.processing:
        q = st.session_state.selected_question
        st.session_state.selected_question = None
        process_question_typing(q, st.session_state.typing_delay)

    if not st.session_state.processing:
        if prompt := st.chat_input("✍️ 궁금한 민원을 입력하세요..."):
            process_question_typing(prompt, st.session_state.typing_delay)

# ---------------------------
# 질문 입력 처리(타자 효과 + 단락간 한 줄 띄기 기능)
# ---------------------------
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
            container = st.empty()
            with st.spinner("🤖 답변 생성 중..."):
                context_text, _, annex_forms = make_context_and_sources(
                    st.session_state.retriever, prompt
                )

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
                    token = getattr(chunk, "content", None)
                    if not token:
                        continue
                    full_text += token
                    container.markdown(full_text)
                    time.sleep(delay)

                # 🚩 자동 한 줄 띄우기 처리 (1번, 2번 처럼 단락 나눔)
                formatted_text = re.sub(r"\n*(\d+\.)", r"\n\n\1", full_text).strip()
                st.session_state.messages.append({"role": "assistant", "content": formatted_text})
                
        except Exception as e:
            err_msg = f"❌ 오류: {e}"
            st.error(err_msg)
            st.session_state.messages.append({"role": "assistant", "content": err_msg})

    st.session_state.processing = False

# ---------------------------
# 푸터
# ---------------------------
def display_footer():
    st.markdown("""
    <div class="foot">
        🏛 곡성군청 | 📞 061-360-0000 | 🌐 www.gokseong.go.kr | 📍 전남 곡성군 곡성읍 군청로 15  
        ⚠ 본 서비스는 AI 안내 서비스이며, 정확한 민원은 담당부서에 문의하세요.
    </div>
    """, unsafe_allow_html=True)

# ---------------------------
# 실행
# ---------------------------
if __name__ == "__main__":
    main()




