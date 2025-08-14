# app.py
import streamlit as st
import os
import time

from rag_logic import (
    prepare_vectorstore,
    build_retriever,
    build_streaming_llm,
    make_context_and_sources,
    build_final_prompt
)

# ---------------------------
# 세션 상태 초기화
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
        "typing_delay": 0.0,  # 즉시 출력
        "model": "gpt-4o-mini",
        "k": 8,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ---------------------------
# 스타일: TOSS 톤 적용 (CSS 주입)
# ---------------------------
def inject_toss_css():
    st.markdown("""
<style>
/* Pretendard 웹폰트 */
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css');

/* Design Tokens */
:root {
  --toss-blue: #3182f6;
  --toss-blue-hover: #1b64da;
  --text-primary: #111827;
  --text-secondary: #6b7280;
  --bg: #f7f8fa;
  --surface: #ffffff;
  --border: #e5e7eb;
  --radius-lg: 12px;
  --radius-md: 8px;
  --radius-xl: 14px;
  --shadow: 0 2px 8px rgba(0,0,0,0.06);
  --success: #10b981;
  --warning: #f59e0b;
  --danger: #ef4444;
}

html, body, [class^="block-container"] {
  font-family: Pretendard, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Inter, system-ui, Apple SD Gothic Neo, Helvetica Neue, Arial, sans-serif;
  color: var(--text-primary);
  background: var(--bg);
  letter-spacing: -0.2px;
}

h1,h2,h3 { font-weight: 700; letter-spacing: -0.3px; margin: 0 0 8px 0; }
p, span, li { color: var(--text-primary); }

/* Header */
.stApp > header { background: var(--bg) !important; border-bottom: 1px solid var(--border); }

/* Sidebar */
section[data-testid="stSidebar"] {
  background: var(--surface);
  border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] [data-testid="stMarkdown"] h1,
section[data-testid="stSidebar"] [data-testid="stMarkdown"] h2 {
  font-size: 16px; font-weight: 700; margin: 12px 0 6px;
}

/* Cards */
.toss-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow);
  padding: 16px 18px;
}

/* Chat bubbles */
.bot-msg, .user-msg {
  border-radius: var(--radius-xl);
  padding: 14px 16px;
  border: 1px solid var(--border);
  box-shadow: var(--shadow);
  animation: fadeInUp .24s ease-out;
  line-height: 1.55;
}
.bot-msg {
  background: var(--surface);
  max-width: 860px;
}
.user-msg {
  background: #f3f4f6;
  margin-left: auto;
  max-width: 720px;
}
.bot-summary {
  color: var(--text-secondary);
  font-size: 13px;
  margin-bottom: 8px;
}

/* Fade */
@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
}

/* Input bar */
.toss-input-wrap {
  position: sticky;
  bottom: 0;
  z-index: 9;
  background: linear-gradient(180deg, rgba(247,248,250,0) 0%, rgba(247,248,250,1) 30%);
  padding-top: 12px;
}
.toss-input {
  display: flex; gap: 8px; align-items: center;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: 8px 10px 8px 12px;
  box-shadow: var(--shadow);
}
.toss-send {
  background: var(--toss-blue); color: #fff; border: none;
  border-radius: 999px; padding: 8px 14px; font-weight: 600;
  transition: background .15s ease;
}
.toss-send:hover { background: var(--toss-blue-hover); }

/* Buttons and chips */
.stButton > button, .toss-btn {
  background: var(--toss-blue) !important;
  color: #fff !important;
  border: 1px solid transparent !important;
  border-radius: var(--radius-md) !important;
  box-shadow: var(--shadow) !important;
}
.stButton > button:hover { background: var(--toss-blue-hover) !important; }
.chip {
  display: inline-block; padding: 6px 10px;
  border: 1px solid var(--border); border-radius: 999px;
  background: #f8fafc; color: var(--text-secondary); font-size: 12px;
  margin-right: 6px; margin-top: 6px;
}
.chip:hover { border-color: #cfd3d9; color: var(--text-primary); }

/* Inputs */
.stTextInput input, .stTextArea textarea {
  border-radius: var(--radius-md);
  border: 1px solid var(--border);
}
.stTextInput input:focus, .stTextArea textarea:focus {
  border-color: var(--toss-blue);
  box-shadow: 0 0 0 3px rgba(49,130,246,0.15);
  outline: none;
}

/* Badges */
.badge {
  display:inline-flex; align-items:center; gap:6px;
  padding: 4px 8px; font-size: 12px; border-radius: 999px; border:1px solid var(--border); background:#fff;
}
.badge-dot { width:8px;height:8px;border-radius:50%; background:#10b981; }

/* Divider */
.divider { height:1px; background: var(--border); margin: 12px 0; }

/* Links in sources */
.source-tag {
  display:inline-block; margin: 4px 6px 0 0; padding: 4px 8px;
  background:#eef2ff; color:#1e40af; font-size:12px; border-radius:999px;
  white-space: nowrap;
}

/* Compact page padding */
.main .block-container { padding-top: 14px; padding-bottom: 80px; }

</style>
""", unsafe_allow_html=True)  # HTML/CSS 렌더 허용[5][12][15]

# ---------------------------
# UI 렌더 함수
# ---------------------------
def render_header():
    index_badge = (
        '<span class="badge"><span class="badge-dot"></span> 인덱스 준비됨</span>'
        if st.session_state.get("index_ready") else
        '<span class="badge" style="border-color:#f59e0b;"><span class="badge-dot" style="background:#f59e0b;"></span> 인덱싱 중</span>'
    )
    st.markdown(f"""
<div class="toss-card" style="display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:12px;">
  <div style="display:flex;align-items:center;gap:10px;">
    <div style="font-size:20px;">🏛️</div>
    <div>
      <div style="font-weight:700;font-size:18px;">곡성군 AI 민원상담봇</div>
      <div style="color:#6b7280;font-size:12px;">곡성군 민원편람 2025 기반 RAG</div>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:10px;">
    {index_badge}
  </div>
</div>
""", unsafe_allow_html=True)

def render_sidebar():
    with st.sidebar:
        st.markdown("#### 최근 질문")
        msgs = st.session_state.get("messages", [])
        shown = 0
        for m in reversed(msgs):
            if m.get("role") == "user":
                st.markdown(
                    f'<div class="toss-card" style="padding:10px;margin-bottom:8px;">{m["content"][:60]}</div>',
                    unsafe_allow_html=True
                )
                shown += 1
                if shown >= 5:
                    break

        st.markdown("#### 자주 묻는 질문")
        presets = ["여권 재발급", "정보공개 청구", "구술·전화 민원", "제증명 수수료"]
        cols = st.columns(2)
        for i, p in enumerate(presets):
            with cols[i % 2]:
                if st.button(p, key=f"preset_{i}"):
                    st.session_state["selected_question"] = p

        st.markdown("---")
        api = st.text_input("OpenAI API Key", type="password")
        if api:
            st.session_state["api_key"] = api

        k = st.number_input("검색 문서 개수(k)", min_value=3, max_value=12, value=st.session_state.get("k", 8), step=1)
        st.session_state["k"] = int(k)

def render_user_msg(text: str):
    st.markdown(f'<div class="user-msg">{text}</div>', unsafe_allow_html=True)

def render_bot_msg(text: str, summary: str = None, sources=None, forms=None):
    if summary:
        st.markdown(f'<div class="bot-msg"><div class="bot-summary">{summary}</div>{text}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-msg">{text}</div>', unsafe_allow_html=True)
    if sources:
        st.markdown('<div style="margin-top:6px;">' +
                    ''.join([f'<span class="source-tag">{s}</span>' for s in sources[:6]]) +
                    '</div>', unsafe_allow_html=True)
    if forms:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("관련 별지/서식", unsafe_allow_html=True)
        st.markdown('<div>' + ''.join([f'<span class="chip">{f}</span>' for f in forms[:8]]) + '</div>', unsafe_allow_html=True)

def render_input_bar():
    st.markdown('<div class="toss-input-wrap">', unsafe_allow_html=True)
    c1, c2 = st.columns([9, 1])
    with c1:
        user_q = st.text_input("질문을 입력하세요", value=st.session_state.get("selected_question") or "", label_visibility="collapsed", key="input_q")
    with c2:
        send = st.button("보내기", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    return user_q, send

# ---------------------------
# 백엔드 유틸
# ---------------------------
def ensure_index():
    if not st.session_state.get("api_key"):
        return False, "API Key가 필요합니다."
    if st.session_state.get("retriever") and st.session_state.get("index_ready"):
        return True, None
    try:
        vectorstore = prepare_vectorstore(
            openai_api_key=st.session_state["api_key"],
            pdf_paths=[st.session_state["pdf_path"]],
            file_names=st.session_state["file_names"],
            vector_dir=st.session_state["vector_dir"],
        )
        retriever = build_retriever(vectorstore, k=st.session_state.get("k", 8))
        st.session_state["retriever"] = retriever
        st.session_state["index_ready"] = True
        return True, None
    except Exception as e:
        return False, f"인덱스 준비 중 오류: {e}"

def generate_answer(question: str):
    # 검색 문맥/소스/서식
    ctx, sources, annex_forms = make_context_and_sources(st.session_state["retriever"], question)
    prompt = build_final_prompt(ctx, question, annex_forms)

    llm = build_streaming_llm(
        model=st.session_state.get("model", "gpt-4o-mini"),
        openai_api_key=st.session_state["api_key"],
        max_tokens=800,
        temperature=0.0,
    )

    # 스트리밍 없이 단순 호출 (streaming=True이지만, 여기선 간단히 invoke)
    try:
        resp = llm.invoke([{"role": "user", "content": prompt}])
        content = getattr(resp, "content", "")
    except Exception as e:
        content = f"답변 생성 중 오류: {e}"
    return content, sources, annex_forms

def extract_summary_from_answer(answer: str, max_lines=3):
    # 간략 요약: 첫 문단/첫 2~3줄
    lines = [l.strip() for l in answer.strip().splitlines() if l.strip()]
    if not lines:
        return None
    snippet = []
    for l in lines:
        snippet.append(l)
        if len(snippet) >= max_lines:
            break
    return " ".join(snippet)

# ---------------------------
# 메인
# ---------------------------
def main():
    init_session_state()
    st.set_page_config(
        page_title="🏛️ 곡성군 AI 민원상담봇",
        page_icon="🏛️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 전역 CSS (TOSS 스타일) 주입
    inject_toss_css()

    # 헤더/사이드바
    render_header()
    render_sidebar()

    # 채팅 기록 렌더
    for m in st.session_state["messages"]:
        if m["role"] == "user":
            render_user_msg(m["content"])
        else:
            render_bot_msg(m["content"], summary=m.get("summary"), sources=m.get("sources"), forms=m.get("forms"))

    # 입력 바
    user_q, send = render_input_bar()

    # 전송 처리
    if send:
        q = (user_q or "").strip()
        if not q:
            st.warning("질문을 입력해주세요.")
            return

        st.session_state["selected_question"] = None
        st.session_state["messages"].append({"role": "user", "content": q})
        render_user_msg(q)

        ok, err = ensure_index()
        if not ok:
            render_bot_msg(f"인덱스 준비 실패: {err}")
            return

        with st.spinner("답변 생성 중…"):
            answer, sources, forms = generate_answer(q)
            summary = extract_summary_from_answer(answer, max_lines=2)
            st.session_state["messages"].append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "forms": forms,
                "summary": summary
            })
            render_bot_msg(answer, summary=summary, sources=sources, forms=forms)

if __name__ == "__main__":
    main()


