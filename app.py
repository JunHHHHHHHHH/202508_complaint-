# app.py
import streamlit as st
import os
import time
import uuid
from rag_logic import (
    prepare_vectorstore,        # 벡터스토어 준비(저장/로드, 해시 비교)
    build_retriever,            # retriever 생성
    build_streaming_l요",
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
        st.session_state.messages.clear()
        st.session_state.question_count = 0
        st.session_state.selected_question = None
        st.session_state.last_clicked_question = None
        st.experimental_rerun()


# ===== 4. 시스템 초기화 (벡터DB 저장/로드) =====
def initialize_system():
    pdf_path = st.session_state.pdf_path
    vector_dir = st.session_state.vector_dir

    if not os.path.exists(pdf_path):
        st.error("❌ 'minweonpyeonram-2025.pdf' 파일이 없습니다.")
        st.stop()

    if not st.session_state.index_ready:
        with st.spinner("📄 인덱스 준비 중입니다... (최초 1회는 시간이 소요될 수 있어요)"):
            vectorstore = prepare_vectorstore(
                openai_api_key=st.session_state.api_key,
                pdf_paths=[pdf_path],
                file_names=st.session_state.file_names,
                vector_dir=vector_dir
            )
            st.session_state.retriever = build_retriever(vectorstore, k=8)
            st.session_state.index_ready = True


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


# ===== 6. 타자기 스타일 순차 출력 + 출처 강화 + 별지서식 =====
def process_question_typing(prompt, delay=0.02):
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

                # 컨텍스트/출처/서식 추출
                context_text, sources_list, annex_forms = make_context_and_sources(
                    st.session_state.retriever, prompt
                )

                # 스트리밍 LLM
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

                container = st.empty()
                full_text = ""
                for chunk in llm.stream(final_prompt):
                    token = getattr(chunk, "content", None)
                    if token is None:
                        token = str(chunk)
                    full_text += token
                    container.markdown(full_text)
                    time.sleep(delay)

                elapsed = round(time.time() - start_time, 2)

                # 근거 출처 모아보기 섹션 추가
                if sources_list:
                    full_text += "\n\n---\n근거 출처 모아보기\n"
                    for i, s in enumerate(sources_list, 1):
                        full_text += f"- {i}. {s}\n"

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


