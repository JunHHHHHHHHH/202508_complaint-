# app.py
import streamlit as st
import os
import tempfile
from rag_logic import initialize_rag_chain, get_answer
import time
import uuid
from datetime import datetime

def init_session_state():
    """세션 상태 변수를 초기화합니다."""
    defaults = {
        "messages": [],
        "rag_chain": None,
        "retriever": None,
        "api_key": None,
        "file_hash": None,
        "file_names": [],
        "chat_id": str(uuid.uuid4()),
        "last_interaction": None,
        "user_feedback": {},
        "question_count": 0
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def main():
    """Streamlit 앱을 실행하는 메인 함수입니다."""
    init_session_state()
    
    st.set_page_config(
        page_title="🏛️ 곡성군 AI민원상담봇",
        page_icon="🏛️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 고급 CSS 스타일링
    st.markdown("""
    <style>
        /* 메인 헤더 스타일 */
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
            padding: 2rem 1rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .main-header h1 {
            font-size: 2.5em;
            font-weight: 700;
            margin-bottom: 0.5rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .main-header p {
            font-size: 1.1em;
            opacity: 0.9;
            margin-bottom: 0;
        }

        /* 메트릭 카드 스타일 */
        .metric-card {
            background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
            padding: 1.5rem;
            border-radius: 15px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
            border-left: 5px solid #667eea;
            margin: 1rem 0;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 35px rgba(0, 0, 0, 0.15);
        }

        .metric-card h4 {
            color: #667eea;
            font-weight: 600;
            margin-bottom: 0.8rem;
            font-size: 1.2em;
        }

        .metric-card p {
            color: #333;
            margin: 0;
            font-size: 1em;
        }

        /* 인사이트 박스 스타일 */
        .insight-box {
            background: linear-gradient(145deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 2rem;
            border-radius: 15px;
            border-left: 5px solid #28a745;
            margin: 1.5rem 0;
            box-shadow: 0 6px 20px rgba(40, 167, 69, 0.1);
        }

        .insight-box h4 {
            color: #28a745;
            font-weight: 600;
            margin-bottom: 1rem;
            font-size: 1.3em;
        }

        .insight-box ul, .insight-box ol {
            color: #555;
            line-height: 1.6;
        }

        .insight-box li {
            margin-bottom: 0.5rem;
        }

        /* 통계 카드 스타일 */
        .stats-card {
            background: linear-gradient(145deg, #fff 0%, #f1f3f4 100%);
            padding: 1.2rem;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
            border: 1px solid #e9ecef;
            margin: 0.5rem 0;
            transition: all 0.3s ease;
        }

        .stats-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.12);
        }

        .stats-number {
            font-size: 2em;
            font-weight: 700;
            color: #667eea;
            margin-bottom: 0.2rem;
        }

        .stats-label {
            color: #6c757d;
            font-size: 0.9em;
            font-weight: 500;
        }

        /* 퀵 액션 버튼 스타일 */
        .quick-action {
            background: linear-gradient(145deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 0.8rem 1.5rem;
            border-radius: 25px;
            border: none;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
            margin: 0.3rem;
            display: inline-block;
            text-decoration: none;
        }

        .quick-action:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        }

        /* 채팅 메시지 스타일 개선 */
        .stChatMessage {
            padding: 1rem;
            border-radius: 15px;
            margin: 0.5rem 0;
        }

        /* 사이드바 스타일 */
        .sidebar-content {
            background: linear-gradient(145deg, #f8f9fa 0%, #ffffff 100%);
            padding: 1rem;
            border-radius: 12px;
            margin: 0.5rem 0;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        }

        /* 푸터 스타일 개선 */
        .footer {
            background: linear-gradient(145deg, #e3f2fd 0%, #bbdefb 100%);
            border: 2px solid #2196f3;
            padding: 2rem;
            border-radius: 20px;
            margin-top: 3rem;
            text-align: center;
            color: #333;
            box-shadow: 0 8px 30px rgba(33, 150, 243, 0.1);
        }

        .footer h4 {
            color: #1976d2;
            margin-bottom: 1rem;
            font-size: 1.4em;
            font-weight: 700;
        }

        .footer p {
            margin: 0.5rem 0;
            color: #555;
            font-weight: 500;
        }

        .footer hr {
            border-color: #2196f3;
            margin: 1.5rem 0;
            opacity: 0.5;
        }

        .footer small {
            color: #666;
            font-style: italic;
            background: rgba(255, 255, 255, 0.8);
            padding: 0.5rem;
            border-radius: 8px;
            display: inline-block;
        }

        /* 로딩 애니메이션 개선 */
        .stSpinner > div {
            border-color: #667eea transparent transparent transparent;
        }

        /* 성공 메시지 스타일 */
        .success-message {
            background: linear-gradient(145deg, #d4edda 0%, #c3e6cb 100%);
            color: #155724;
            padding: 1rem;
            border-radius: 10px;
            border-left: 4px solid #28a745;
            margin: 1rem 0;
        }

        /* 경고 메시지 스타일 */
        .warning-message {
            background: linear-gradient(145deg, #fff3cd 0%, #ffeaa7 100%);
            color: #856404;
            padding: 1rem;
            border-radius: 10px;
            border-left: 4px solid #ffc107;
            margin: 1rem 0;
        }

        /* 애니메이션 효과 */
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translate3d(0, 40px, 0);
            }
            to {
                opacity: 1;
                transform: translate3d(0, 0, 0);
            }
        }

        .fade-in-up {
            animation: fadeInUp 0.6s ease-out;
        }

        /* 반응형 디자인 */
        @media (max-width: 768px) {
            .main-header h1 {
                font-size: 1.8em;
            }
            .main-header p {
                font-size: 1em;
            }
            .metric-card, .insight-box {
                padding: 1rem;
            }
        }
    </style>
    """, unsafe_allow_html=True)

    # 메인 헤더
    st.markdown("""
    <div class="main-header fade-in-up">
        <h1>🏛️ 곡성군 AI민원상담봇</h1>
        <p>AI 기반 민원업무 구비서류, 처리기간, 처리흐름 을 쉽고 빠르게 안내해드립니다</p>
    </div>
    """, unsafe_allow_html=True)

    # 사이드바 설정
    setup_sidebar()

    # 메인 컨텐츠 영역
    if not st.session_state.get("api_key"):
        display_api_key_warning()
        return

    # 시스템 초기화
    initialize_system()

    # 메인 탭 구성
    tab1, tab2, tab3 = st.tabs(["💬 AI 상담", "📊 사용 통계", "ℹ️ 서비스 정보"])
    
    with tab1:
        display_chat_interface()
    
    with tab2:
        display_usage_stats()
    
    with tab3:
        display_service_info()

    # 푸터
    display_footer()

def setup_sidebar():
    """사이드바를 설정합니다."""
    st.sidebar.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    
    # API 키 입력
    st.sidebar.title("🔑 시스템 설정")
    api_key = st.sidebar.text_input(
        "OpenAI API 키",
        type="password",
        placeholder="sk-...",
        help="OpenAI API 키가 필요합니다.",
        key="api_key_input"
    )
    
    if api_key:
        st.session_state.api_key = api_key
        st.sidebar.success("✅ API 키가 설정되었습니다")
    
    st.sidebar.markdown("---")
    
    # 빠른 질문 템플릿
    st.sidebar.title("🚀 주요 민원 질문")
    
    quick_questions = [
        "여권 발급 절차는?",
        "주민등록등본 발급 절차는?",
        "인감증명서 발급 절차는?",
        "정보공개 청구 방법은?",
        "건축허가 신청 시 필요한 서류는?"
    ]
    
    for i, question in enumerate(quick_questions):
        if st.sidebar.button(
            question, 
            key=f"quick_q_{i}",
            help=f"클릭하면 '{question}' 질문이 자동으로 입력됩니다"
        ):
            st.session_state.selected_question = question
            st.rerun()
    
    st.sidebar.markdown("---")
    
    # 채팅 초기화 버튼
    if st.sidebar.button("🗑️ 대화 기록 초기화", type="secondary"):
        st.session_state.messages = []
        st.session_state.chat_id = str(uuid.uuid4())
        st.success("대화 기록이 초기화되었습니다.")
        st.rerun()
    
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

def display_api_key_warning():
    """API 키 입력 경고를 표시합니다."""
    st.markdown("""
    <div class="warning-message fade-in-up">
        <h3>⚠️ API 키가 필요합니다</h3>
        <p>사이드바에서 OpenAI API 키를 입력해주세요.</p>
        <p><strong>API 키 발급:</strong> <a href="https://platform.openai.com/api-keys" target="_blank">OpenAI Platform</a></p>
    </div>
    """, unsafe_allow_html=True)

def initialize_system():
    """시스템을 초기화합니다."""
    default_pdf_path = "minweonpyeonram-2025.pdf"
    
    if not os.path.exists(default_pdf_path):
        st.error("❌ 곡성군 민원편람 파일을 찾을 수 없습니다.")
        st.info("💡 'minweonpyeonram-2025.pdf' 파일이 필요합니다.")
        return

    pdf_files = [default_pdf_path]
    file_names = ["곡성군 민원편람 2025"]
    file_hash = str(hash(open(default_pdf_path, 'rb').read()))
    
    if (not st.session_state.get("rag_chain") or
        st.session_state.get("file_hash") != file_hash):
        
        try:
            with st.spinner("🔄 곡성군 민원편람을 분석하고 있습니다..."):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                rag_chain, retriever, api_key = initialize_rag_chain(
                    st.session_state.api_key, pdf_files, file_names
                )
                
                st.session_state.rag_chain = rag_chain
                st.session_state.retriever = retriever
                st.session_state.file_hash = file_hash
                st.session_state.file_names = file_names
                
                st.success("✅ 곡성군 민원편람 분석 완료!")
                
        except Exception as e:
            st.error(f"❌ 시스템 초기화 오류: {str(e)}")
            return

def display_chat_interface():
    """채팅 인터페이스를 표시합니다."""
    # 현재 분석 문서 표시
    st.markdown(f"""
    <div class="metric-card fade-in-up">
        <h4>📖 현재 상담 가능 문서</h4>
        <p><strong>{', '.join(st.session_state.file_names)}</strong></p>
        <p style="color: #6c757d; font-size: 0.9em; margin-top: 0.5rem;">
            총 질문 수: {st.session_state.question_count}개
        </p>
    </div>
    """, unsafe_allow_html=True)

    
    # 사용 안내
    with st.expander("📖 사용 안내", expanded=False):
        st.markdown("#### 🎯 이용 방법")
        st.markdown("""
        1. **사이드바 빠른 질문**: 자주 묻는 질문을 클릭하세요
        2. **직접 질문**: 아래 채팅창에 궁금한 민원업무를 입력하세요
        3. **구체적 질문**: "○○ 신청 방법", "○○ 필요서류", "○○ 처리기간" 등
        """)
    
        st.markdown("#### 💡 질문 예시")
        st.markdown("""
        - "여권 발급은 어떻게 하나요?"
        - "정보공개 청구 시 필요한 서류와 처리기간을 알려주세요"
        - "주민등록 관련 업무는 무엇이 있나요?"
        - "온라인으로 신청할 수 있는 민원이 있나요?"
        """)
        

    # 채팅 기록 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 환영 메시지
    if not st.session_state.messages:
        welcome_message = """
안녕하세요! 🙋‍♀️ 곡성군 민원편람 AI 상담봇입니다.

**무엇을 도와드릴까요?**
- 민원업무 처리절차 안내
- 필요서류 및 구비사항 확인  
- 처리기간 및 수수료 안내
- 신청방법 및 접수처 정보
- 관련 서식 안내

궁금한 민원업무를 말씀해 주시면 자세히 안내해드리겠습니다! 😊
        """
        st.session_state.messages.append({"role": "assistant", "content": welcome_message})

    # 채팅 기록 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 선택된 빠른 질문 처리
    if st.session_state.get("selected_question"):
        process_question(st.session_state.selected_question)
        st.session_state.selected_question = None

    # 채팅 입력
    if prompt := st.chat_input("민원업무에 대해 궁금한 점을 입력하세요..."):
        process_question(prompt)

def process_question(prompt):
    """질문을 처리합니다."""
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.question_count += 1
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        try:
            with st.spinner("답변을 생성하고 있습니다..."):
                start_time = time.time()
                
                response = get_answer(
                    st.session_state.rag_chain,
                    st.session_state.retriever,
                    prompt,
                    st.session_state.api_key
                )
                
                end_time = time.time()
                response_time = round(end_time - start_time, 2)
                
                # 응답 시간 표시
                response_with_time = f"{response}\n\n---\n⏱️ *응답 시간: {response_time}초*"
                
                st.markdown(response_with_time)
                st.session_state.messages.append({"role": "assistant", "content": response_with_time})
                
                # 피드백 버튼
                col1, col2, col3 = st.columns([1, 1, 3])
                with col1:
                    if st.button("👍 도움됨", key=f"like_{len(st.session_state.messages)}"):
                        st.session_state.user_feedback[len(st.session_state.messages)] = "positive"
                        st.success("피드백 감사합니다!")
                
                with col2:
                    if st.button("👎 개선필요", key=f"dislike_{len(st.session_state.messages)}"):
                        st.session_state.user_feedback[len(st.session_state.messages)] = "negative"
                        st.info("피드백이 기록되었습니다.")
                
        except Exception as e:
            error_msg = f"❌ 답변 생성 중 오류가 발생했습니다: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

def display_usage_stats():
    """사용 통계를 표시합니다."""
    st.subheader("📊 사용 통계 대시보드")
    
    # 통계 카드들
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-number">{st.session_state.question_count}</div>
            <div class="stats-label">총 질문 수</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        message_count = len([m for m in st.session_state.messages if m["role"] == "assistant"]) - 1
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-number">{message_count}</div>
            <div class="stats-label">답변 수</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        positive_feedback = len([f for f in st.session_state.user_feedback.values() if f == "positive"])
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-number">{positive_feedback}</div>
            <div class="stats-label">긍정 피드백</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        negative_feedback = len([f for f in st.session_state.user_feedback.values() if f == "negative"])
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-number">{negative_feedback}</div>
            <div class="stats-label">개선 요청</div>
        </div>
        """, unsafe_allow_html=True)
    
    # 만족도 차트
    if st.session_state.user_feedback:
        st.subheader("📈 사용자 만족도")
        feedback_data = {"긍정": positive_feedback, "부정": negative_feedback}
        
        # 간단한 막대 차트 표시
        col1, col2 = st.columns(2)
        with col1:
            st.metric("긍정 피드백", positive_feedback, delta=None)
        with col2:
            st.metric("부정 피드백", negative_feedback, delta=None)

def display_service_info():
    """서비스 정보를 표시합니다."""
    st.subheader("ℹ️ 서비스 정보")
    
    st.markdown("""
    <div class="insight-box">
        <h4>🏛️ 곡성군 민원편람 AI 상담봇</h4>
        <p><strong>버전:</strong> 2.0.0</p>
        <p><strong>최종 업데이트:</strong> 2025년 8월</p>
        <p><strong>지원 문서:</strong> 곡성군 민원편람 2025</p>
        
        <h4>🔧 주요 기능</h4>
        <ul>
            <li>민원업무 처리절차 안내</li>
            <li>구비서류 및 서식 정보 제공</li>
            <li>처리기간 및 수수료 안내</li>
            <li>담당부서 및 연락처 정보</li>
            <li>실시간 AI 기반 상담</li>
        </ul>
        
        <h4>⚡ 기술 스택</h4>
        <ul>
            <li>Frontend: Streamlit</li>
            <li>AI Model: GPT-4o-mini</li>
            <li>Vector Database: FAISS</li>
            <li>Embeddings: OpenAI text-embedding-3-small</li>
            <li>Framework: LangChain</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def display_footer():
    """푸터를 표시합니다."""
    st.markdown("""
    <div class="footer fade-in-up">
        <h4>🏛️ 곡성군청</h4>
        <p>📞 대표전화: 061-360-0000 | 🌐 www.gokseong.go.kr</p>
        <p>📍 전라남도 곡성군 곡성읍 군청로 15</p>
        <hr>
        <small>⚠️ 본 서비스는 AI 기반 안내서비스로, 정확한 민원처리를 위해서는 담당부서에 직접 문의하시기 바랍니다.</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()



