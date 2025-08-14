# app.py
import streamlit as st
import os
import tempfile
from rag_logic import initialize_rag_chain, get_answer

# 페이지 설정
st.set_page_config(
    page_title="곡성군 민원편람 챗봇", 
    page_icon="🏛️",
    layout="wide"
)

# 사이드바 스타일링
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f77b4, #17becf);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .quick-question-btn {
        margin: 5px;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 5px;
        background-color: #f8f9fa;
        cursor: pointer;
    }
    .footer {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin-top: 2rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# 메인 헤더
st.markdown("""
<div class="main-header">
    <h1>🏛️ 곡성군 민원편람 AI 상담봇</h1>
    <p>민원업무 처리기간, 구비서류, 처리흐름을 쉽고 빠르게 안내해드립니다</p>
</div>
""", unsafe_allow_html=True)

# OpenAI API 키 입력
st.sidebar.title("🔑 API 설정")
openai_api_key = st.sidebar.text_input(
    "OpenAI API 키를 입력하세요:",
    type="password",
    placeholder="sk-...",
    help="OpenAI API 키가 필요합니다. https://platform.openai.com/api-keys 에서 발급받으세요."
)

if not openai_api_key:
    st.warning("⚠️ 사이드바에서 OpenAI API 키를 입력해주세요.")
    st.info("💡 API 키 발급: https://platform.openai.com/api-keys")
    st.stop()

# 빠른 질문 템플릿
st.sidebar.title("🚀 빠른 질문")
quick_questions = {
    "📋 처리기간 관련": [
        "여권 발급 처리기간이 얼마나 걸리나요?",
        "정보공개 청구 처리기간은?",
        "주민등록 등초본 발급은 얼마나 걸리나요?"
    ],
    "📄 구비서류 관련": [
        "여권 재발급 시 필요한 서류는?",
        "정보공개 청구 시 필요한 서류는?",
        "인감증명서 발급에 필요한 서류는?"
    ],
    "💰 수수료 관련": [
        "여권 발급 수수료는 얼마인가요?",
        "각종 증명서 발급 비용이 궁금해요",
        "취득세 신고 관련 수수료는?"
    ],
    "🏢 민원 접수 관련": [
        "민원은 어떻게 신청하나요?",
        "온라인으로 신청 가능한 민원이 있나요?",
        "방문 없이 처리할 수 있는 업무는?"
    ]
}

for category, questions in quick_questions.items():
    st.sidebar.subheader(category)
    for question in questions:
        if st.sidebar.button(question, key=f"btn_{question}"):
            st.session_state.selected_question = question

# 기본 PDF 파일 경로 설정
default_pdf_path = "minweonpyeonram-2025.pdf"

# PDF 파일 확인 및 처리
if os.path.exists(default_pdf_path):
    pdf_files = [default_pdf_path]
    file_names = ["곡성군 민원편람 2025"]
    
    # RAG 체인 초기화
    file_hash = str(hash(open(default_pdf_path, 'rb').read()))
    
    if ("rag_chain" not in st.session_state or
        st.session_state.get("api_key") != openai_api_key or
        st.session_state.get("file_hash") != file_hash):
        
        try:
            with st.spinner("🔄 곡성군 민원편람을 분석하고 있습니다..."):
                rag_chain, retriever, api_key = initialize_rag_chain(
                    openai_api_key, pdf_files, file_names
                )
                
                st.session_state.rag_chain = rag_chain
                st.session_state.retriever = retriever
                st.session_state.api_key = api_key
                st.session_state.file_hash = file_hash
                st.session_state.file_names = file_names
                
                st.success("✅ 곡성군 민원편람 분석 완료!")
                
        except Exception as e:
            st.error(f"❌ 시스템 초기화 오류: {str(e)}")
            st.info("💡 관리자에게 문의해주세요.")
            st.stop()
else:
    st.error("❌ 곡성군 민원편람 파일을 찾을 수 없습니다.")
    st.info("💡 'minweonpyeonram-2025.pdf' 파일이 필요합니다.")
    st.stop()

# 현재 분석 중인 문서 표시
st.info(f"📖 현재 상담 가능 문서: **{', '.join(st.session_state.file_names)}**")

# 사용 안내
with st.expander("📖 사용 안내", expanded=False):
    st.markdown("""
    ### 🎯 이용 방법
    1. **사이드바 빠른 질문**: 자주 묻는 질문을 클릭하세요
    2. **직접 질문**: 아래 채팅창에 궁금한 민원업무를 입력하세요
    3. **구체적 질문**: "○○ 신청 방법", "○○ 필요서류", "○○ 처리기간" 등
    
    ### 💡 질문 예시
    - "여권 발급은 어떻게 하나요?"
    - "정보공개 청구 시 필요한 서류와 처리기간을 알려주세요"
    - "주민등록 관련 업무는 무엇이 있나요?"
    - "온라인으로 신청할 수 있는 민원이 있나요?"
    """)

# 채팅 기록 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 환영 메시지 (처음 방문시에만)
if not st.session_state.messages:
    welcome_message = """
안녕하세요! 🙋‍♀️ 곡성군 민원편람 AI 상담봇입니다.

**무엇을 도와드릴까요?**
- 민원업무 처리절차 안내
- 필요서류 및 구비사항 확인  
- 처리기간 및 수수료 안내
- 신청방법 및 접수처 정보

궁금한 민원업무를 말씀해 주시면 자세히 안내해드리겠습니다! 😊
    """
    st.session_state.messages.append({"role": "assistant", "content": welcome_message})

# 채팅 기록 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 선택된 빠른 질문 처리
if "selected_question" in st.session_state:
    prompt = st.session_state.selected_question
    del st.session_state.selected_question
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        try:
            with st.spinner("답변을 생성하고 있습니다..."):
                response = get_answer(
                    st.session_state.rag_chain,
                    st.session_state.retriever,
                    prompt,
                    st.session_state.api_key
                )
                
                # 답변에 추가 정보 포함
                enhanced_response = f"{response}\n\n---\n💬 **추가 문의**: 곡성군청 ☎ 061-360-0000\n🌐 **온라인**: 곡성군 홈페이지 또는 정부24"
                
                st.markdown(enhanced_response)
                st.session_state.messages.append({"role": "assistant", "content": enhanced_response})
                
        except Exception as e:
            error_msg = f"❌ 답변 생성 중 오류가 발생했습니다: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

# 채팅 입력
if prompt := st.chat_input("민원업무에 대해 궁금한 점을 입력하세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("답변을 생성하고 있습니다..."):
                response = get_answer(
                    st.session_state.rag_chain,
                    st.session_state.retriever,
                    prompt,
                    st.session_state.api_key
                )
                
                # 답변에 추가 정보 포함
                enhanced_response = f"{response}\n\n---\n💬 **추가 문의**: 곡성군청 ☎ 061-360-0000\n🌐 **온라인**: 곡성군 홈페이지 또는 정부24"
                
                st.markdown(enhanced_response)
                st.session_state.messages.append({"role": "assistant", "content": enhanced_response})
                
        except Exception as e:
            error_msg = f"❌ 답변 생성 중 오류가 발생했습니다: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

# 피드백 시스템
st.sidebar.markdown("---")
st.sidebar.subheader("📝 서비스 평가")
col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("👍 도움되었어요"):
        st.sidebar.success("피드백 감사합니다!")

with col2:
    if st.button("👎 개선필요"):
        feedback = st.sidebar.text_area("개선사항을 알려주세요:", height=100)
        if st.sidebar.button("의견 제출"):
            st.sidebar.info("소중한 의견 감사합니다!")

# 푸터
st.markdown("""
<div class="footer">
    <h4>🏛️ 곡성군청</h4>
    <p>📞 대표전화: 061-360-0000 | 🌐 www.gokseong.go.kr</p>
    <p>📍 전라남도 곡성군 곡성읍 군청로 15</p>
    <hr>
    <small>⚠️ 본 서비스는 AI 기반 안내서비스로, 정확한 민원처리를 위해서는 담당부서에 직접 문의하시기 바랍니다.</small>
</div>
""", unsafe_allow_html=True)
