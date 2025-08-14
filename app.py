# app.py
import streamlit as st
import os

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

def render_hero():
    st.markdown(THEME_CSS, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="hero">
      <div class="hero-eyebrow">🏛️ 곡성군 AI 민원상담봇</div>
      <div class="hero-title">민원, 더 간결하고 빠르게 해결해요.</div>
      <div class="hero-desc">곡성군 민원편람 기반으로 답변 드립니다.</div>
    </div>
    """, unsafe_allow_html=True)

setup_sidebar()

    if not st.session_state.api_key:
        st.warning("🔑 사이드바에서 OpenAI API 키를 입력해주세요.")
        st.stop()

    initialize_system()
    display_chat_interface()
    display_footer()


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



def answer_question(question: str):
    api_key = st.session_state.get("api_key") or os.getenv("OPENAI_API_KEY")
    llm = build_streaming_llm(model="gpt-4o-mini", openai_api_key=api_key, max_tokens=800, temperature=0)
    context, sources, annex = make_context_and_sources(st.session_state["retriever"], question)
    prompt = build_final_prompt(context, question, annex)
    resp = llm.invoke(prompt)
    return resp.content, sources, annex

def render_chat_ui():
    st.markdown('<div class="card"><h3>📝 민원 질문</h3>', unsafe_allow_html=True)
    col1, col2 = st.columns([4,1])
    with col1:
        q = st.text_input("궁금하신 사항항을 입력하세요", label_visibility="collapsed")
    with col2:
        ask = st.button("바로 확인")
    st.markdown('</div>', unsafe_allow_html=True)

    if ask and q.strip():
        ok, err = ensure_retriever()
        if not ok:
            st.error(err)
            return
        with st.spinner("민원 정보를 정리하는 중..."):
            answer, sources, annex = answer_question(q.strip())
        st.markdown('<div class="card"><h3>🔎 결과</h3>', unsafe_allow_html=True)
        st.markdown(f'<div class="msg-bot">{answer}</div>', unsafe_allow_html=True)
        with st.expander("출처 보기"):
            for s in sources:
                st.markdown(f"- {s}")
        if annex:
            with st.expander("관련 별지/서식"):
                for a in annex:
                    st.markdown(f"- {a}")
        st.markdown('</div>', unsafe_allow_html=True)

def main():
    init_session_state()
    st.set_page_config(page_title="🏛️ 곡성군 AI 민원상담봇", page_icon="🏛️", layout="wide", initial_sidebar_state="collapsed")
    with st.sidebar:
        st.header("설정")
        api = st.text_input("OpenAI API Key", type="password")
        if api:
            st.session_state["api_key"] = api
        st.caption("PDF: minweonpyeonram-2025.pdf")
        if st.button("색인 재생성"):
            st.session_state["retriever"] = None
            st.session_state["index_ready"] = False
            st.success("다음 질문 시 자동 재색인합니다.")
    render_hero()
    render_quick_pills()
    render_chat_ui()
    st.markdown('<div class="foot">© Gokseong-gun · 민원편람 기반 안내</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()




