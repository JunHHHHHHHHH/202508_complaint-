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
# CSS 테마 + 카드 스타일
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
.msg-bot { line-height: 1.6; }
.card-box {
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 12px 16px;
  margin-bottom: 12px;
  background: var(--card-bg);
}
.card-title {
  font-weight: 700;
  color: var(--accent-2);
  margin-bottom: 6px;
}
.card-content {
  margin-left: 0.5em;
}
.msg-indent { margin-left: 1.2em; }
.step-num {
  color: var(--accent);
  font-weight: bold;
}
</style>
"""

# ---------------------------
def init_session_state():
    defaults = {
        "messages": [], "api_key": None, "question_count": 0,
        "processing": False, "selected_question": None,
        "last_clicked_question": None, "vector_dir": "faiss_minweonpyeonram_2025",
        "pdf_path": "minweonpyeonram-2025.pdf", "index_ready": False,
        "retriever": None, "file_names": ["곡성군 민원편람 2025"], "typing_delay": 0.02
    }
    for k,v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

def render_hero():
    st.markdown(THEME_CSS, unsafe_allow_html=True)
    st.markdown("""
    <div class="hero">
      <div class="hero-eyebrow">🏛️ 곡성군 AI 민원상담봇</div>
      <div class="hero-title">민원, 더 간결하고 빠르게 해결해요.</div>
      <div class="hero-desc">곡성군 민원편람 기반으로 답변 드립니다.</div>
    </div>
    """, unsafe_allow_html=True)

def main():
    init_session_state()
    st.set_page_config(page_title="곡성군 AI 민원상담봇", page_icon="🏛️", layout="wide")
    render_hero()
    setup_sidebar()
    if not st.session_state.api_key:
        st.warning("🔑 사이드바에서 OpenAI API 키를 입력해주세요.")
        st.stop()
    initialize_system()
    display_chat_interface()
    display_footer()

def setup_sidebar():
    st.sidebar.title("API 설정")
    key = st.sidebar.text_input("OpenAI API 키", type="password")
    if key: st.session_state.api_key = key
    st.sidebar.markdown("---")
    st.sidebar.subheader("빠른 질문")
    qs = [
        "여권을 발급 받고 싶어요","전입신고 방법을 알고 싶어요",
        "인감증명서를 발급 받고 싶어요","정보공개를 청구방법을 알고 싶어요",
        "건축허가 신청 절차를 알고 싶어요"
    ]
    for q in qs:
        if st.sidebar.button(q):
            if not st.session_state.processing: 
                st.session_state.selected_question = q
    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ 초기화"):
        st.session_state.clear()
        init_session_state()
        st.experimental_rerun()

def initialize_system():
    if not os.path.exists(st.session_state.pdf_path):
        st.error(f"파일 없음: {st.session_state.pdf_path}")
        st.stop()
    if not st.session_state.index_ready:
        with st.spinner("📄 인덱스 준비 중..."):
            vs = prepare_vectorstore(st.session_state.api_key,
                                     [st.session_state.pdf_path],
                                     st.session_state.file_names,
                                     st.session_state.vector_dir)
            st.session_state.retriever = build_retriever(vs, k=8)
            st.session_state.index_ready=True

def display_chat_interface():
    st.markdown(f"<div class='card'>문서: {', '.join(st.session_state.file_names)} | 질문: {st.session_state.question_count}</div>", unsafe_allow_html=True)
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"], unsafe_allow_html=True)
    if st.session_state.selected_question and not st.session_state.processing:
        process_question_typing(st.session_state.selected_question)
        st.session_state.selected_question=None
    if not st.session_state.processing:
        if p := st.chat_input("✍️ 민원을 입력하세요..."):
            process_question_typing(p)

def format_as_cards(text:str):
    sections = {
        "민원업무명": "", "처리기간":"", "구비서류":"", "수수료":"", "처리 절차":""
    }
    for key in sections.keys():
        pattern = fr"{key}\s*:(.+?)(?=\n\n|$)"
        m = re.search(pattern, text, re.S)
        if m:
            content = m.group(1).strip()
            if key=="구비서류":
                items = [f"<div class='msg-indent'>• {i.strip()}</div>" for i in content.split("\n") if i.strip()]
                content = "\n".join(items)
            if key=="처리 절차":
                content = re.sub(r"(\d+\s*단계\s*:)", r"<span class='step-num'>\1</span>", content)
                steps = [f"<div class='msg-indent'>{line.strip()}</div>" for line in content.split("\n") if line.strip()]
                content = "\n".join(steps)
            sections[key]=content
    html=""
    for k,v in sections.items():
        if v:
            html += f"<div class='card-box'><div class='card-title'>{k}</div><div class='card-content'>{v}</div></div>"
    return html

def process_question_typing(prompt, delay=0.02):
    if st.session_state.processing: return
    st.session_state.processing=True
    st.session_state.messages.append({"role":"user","content":prompt})
    st.session_state.question_count+=1
    with st.chat_message("user"): st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            container=st.empty()
            with st.spinner("🤖 답변 생성 중..."):
                ctx,_,af = make_context_and_sources(st.session_state.retriever, prompt)
                llm=build_streaming_llm("gpt-4o-mini", st.session_state.api_key, max_tokens=800, temperature=0)
                fp = build_final_prompt(ctx, prompt, af)
                full_text=""
                for chunk in llm.stream(fp):
                    token=getattr(chunk,"content",None)
                    if token:
                        full_text += token
                        container.markdown(full_text)
                        time.sleep(delay)
                # 번호 줄바꿈
                formatted = re.sub(r"\n*(\d+\.)", r"\n\n\1", full_text).strip()
                # 카드 변환
                card_html = format_as_cards(formatted)
                st.session_state.messages.append({"role":"assistant","content":f"<div class='msg-bot'>{card_html}</div>"})
        except Exception as e:
            st.error(e)
            st.session_state.messages.append({"role":"assistant","content":str(e)})
    st.session_state.processing=False

def display_footer():
    st.markdown("<div class='foot'>🏛 곡성군청 | 📞 061-360-0000 | 🌐 www.gokseong.go.kr<br>⚠ AI 안내, 정확한 민원은 담당부서 문의</div>", unsafe_allow_html=True)

if __name__=="__main__":
    main()
