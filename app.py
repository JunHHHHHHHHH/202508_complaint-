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
# CSS 테마 (Buybrand 스타일 톤 앤 매너 + 답변 가독성 강화)
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
.msg-bot p {
  line-height: 1.6;
  margin-bottom: 0.6rem;
}
.msg-bot strong {
  color: var(--accent);
}
.msg-indent {
  margin-left: 1.2em;
}
.step-num {
  color: var(--accent-2);
  font-weight: bold;
}
</style>
"""

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
        "typing_delay": 0.02
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def render_hero():
    st.markdown(THEME_CSS, unsafe_allow_html=True)
    st.markdown("""
    <div class="hero">
      <div class="hero-eyebrow">🏛️ 곡성군 AI 민원상담봇</div>
      <div class="hero-title">민원, 보기 좋게 정리해드립니다.</div>
      <div class="hero-desc">목록과 절차를 읽기 쉽게 들여쓰기 + 색상 강조 적용</div>
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
    key = st.sidebar.text_input("OpenAI API 키", type="password", key="api_key_input")
    if key: st.session_state.api_key = key

    st.sidebar.markdown("---")
    st.sidebar.subheader("빠른 질문")
    for q in ["여권을 발급 받고 싶어요","전입신고 방법을 알고 싶어요","인감증명서 발급 받고 싶어요","정보공개를 청구방법을 알고 싶어요","건축허가 신청 절차를 알고 싶어요"]:
        if st.sidebar.button(q):
            if not st.session_state.processing and st.session_state.last_clicked_question != q:
                st.session_state.selected_question, st.session_state.last_clicked_question = q, q

    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ 대화 초기화"):
        st.session_state.clear()
        init_session_state()
        st.experimental_rerun()

def initialize_system():
    if not os.path.exists(st.session_state.pdf_path):
        st.error(f"❌ '{st.session_state.pdf_path}' 파일이 없습니다.")
        st.stop()
    if not st.session_state.index_ready:
        with st.spinner("📄 인덱스 준비 중..."):
            vectorstore = prepare_vectorstore(
                st.session_state.api_key,
                [st.session_state.pdf_path],
                st.session_state.file_names,
                st.session_state.vector_dir
            )
            st.session_state.retriever = build_retriever(vectorstore, k=8)
            st.session_state.index_ready = True

def display_chat_interface():
    st.markdown(
        f"<div class='card'>📄 문서: <b>{', '.join(st.session_state.file_names)}</b> | 💬 질문 수: {st.session_state.question_count}</div>",
        unsafe_allow_html=True
    )
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"], unsafe_allow_html=True)

    if st.session_state.selected_question and not st.session_state.processing:
        process_question_typing(st.session_state.selected_question)
        st.session_state.selected_question = None

    if not st.session_state.processing:
        if prompt := st.chat_input("✍️ 민원을 입력하세요..."):
            process_question_typing(prompt)

def process_question_typing(prompt, delay=0.02):
    if st.session_state.processing: return
    if st.session_state.messages and st.session_state.messages[-1]["role"]=="user" and st.session_state.messages[-1]["content"]==prompt:
        return
    st.session_state.processing = True
    st.session_state.messages.append({"role":"user","content":prompt})
    st.session_state.question_count += 1

    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            container = st.empty()
            with st.spinner("🤖 답변 생성 중..."):
                ctx, _, annex = make_context_and_sources(st.session_state.retriever, prompt)
                llm = build_streaming_llm("gpt-4o-mini", st.session_state.api_key, max_tokens=800, temperature=0)
                full_prompt = build_final_prompt(ctx, prompt, annex)

                full_text = ""
                for chunk in llm.stream(full_prompt):
                    token = getattr(chunk, "content", None)
                    if token:
                        full_text += token
                        container.markdown(full_text)
                        time.sleep(delay)

                # 📌 1) 번호 앞 줄바꿈
                formatted_text = re.sub(r"\n*(\d+\.)", r"\n\n\1", full_text).strip()

                # 📌 2) 주요 제목 굵게 + 컬러
                keywords = ["민원업무명", "처리기간", "구비서류", "수수료", "처리 절차"]
                for kw in keywords:
                    formatted_text = re.sub(fr"\n*({kw}\s*:)", rf"\n\n**<span style='color:#A6E3A1'>\1</span>**", formatted_text)

                # 📌 3) 구비서류 목록 불릿 & 들여쓰기
                formatted_text = re.sub(r"(구비서류\s*:\s*)(.+?)(?=(\n\n|$))",
                                        lambda m: m.group(1) + "\n" +
                                                  "\n".join([f"<span class='msg-indent'>• {item.strip()}</span>"
                                                              for item in m.group(2).split("\n") if item.strip()]),
                                        formatted_text, flags=re.S)

                # 📌 4) 처리 절차 단계 (1단계:, 2단계:) 색상 강조 + 들여쓰기
                formatted_text = re.sub(r"(\d+\s*단계\s*:)",
                                        r"<span class='step-num'>\1</span>",
                                        formatted_text)

                st.session_state.messages.append({"role":"assistant",
                                                  "content":f"<div class='msg-bot'>{formatted_text}</div>"})

        except Exception as e:
            st.error(f"❌ 오류: {e}")
            st.session_state.messages.append({"role":"assistant","content":str(e)})

    st.session_state.processing = False

def display_footer():
    st.markdown("""
    <div class="foot">
        🏛 곡성군청 | 📞 061-360-0000 | 🌐 www.gokseong.go.kr | 📍 전남 곡성군 곡성읍 군청로 15  
        ⚠ AI 안내 서비스이며, 정확한 민원은 담당부서에 문의하세요.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()




