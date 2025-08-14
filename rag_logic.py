# rag_logic.py

import os
import hashlib
import json
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from dotenv import load_dotenv

# --- 설정값 ---
PDF_PATH = "minweonpyeonram-2025.pdf"
FAISS_INDEX_PATH = "./faiss_index_minweon" # 벡터 DB를 저장할 폴더
HASH_FILE_PATH = os.path.join(FAISS_INDEX_PATH, "pdf_hash.json") # PDF 파일 변경 감지를 위한 해시 저장 파일

def get_pdf_hash(file_path):
    """PDF 파일의 해시(SHA-256) 값을 계산하여 파일 변경 여부를 확인합니다."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

# Streamlit의 캐시 기능을 사용해 벡터 DB와 RAG 체인을 한번만 로드합니다.
# 이렇게 하면 앱을 새로고침해도 매번 새로 만들지 않아 매우 빠릅니다.
@st.cache_resource(show_spinner="AI 민원 상담봇을 준비 중입니다...")
def initialize_rag_chain():
    """
    RAG 체인을 초기화하는 함수.
    1. PDF 파일의 변경 여부를 확인합니다.
    2. 변경되지 않았으면 저장된 벡터 DB를 로드합니다.
    3. 변경되었거나 새로 만들 경우, PDF를 읽어 벡터 DB를 생성하고 저장합니다.
    4. 어르신 눈높이에 맞춘 프롬프트와 함께 RAG 체인을 생성하여 반환합니다.
    """
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OpenAI API 키가 설정되지 않았습니다. .env 파일에 키를 추가해주세요.")
        return None, None

    # PDF 파일 존재 여부 확인
    if not os.path.exists(PDF_PATH):
        st.error(f"'{PDF_PATH}' 파일을 찾을 수 없습니다. 프로젝트 폴더에 PDF 파일을 넣어주세요.")
        return None, None

    # 벡터 DB 저장 폴더 생성
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)

    current_pdf_hash = get_pdf_hash(PDF_PATH)
    stored_hash = None

    # 저장된 해시 값 읽기
    if os.path.exists(HASH_FILE_PATH):
        with open(HASH_FILE_PATH, 'r') as f:
            stored_hash = json.load(f).get('hash')

    embeddings = OpenAIEmbeddings()

    # PDF가 변경되지 않았다면 기존 벡터 DB 로드
    if os.path.exists(os.path.join(FAISS_INDEX_PATH, "index.faiss")) and current_pdf_hash == stored_hash:
        st.info("기존에 저장된 민원편람 데이터를 불러옵니다.")
        vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True  # 사용자 신뢰 기반 로드 허용
        )
    # PDF가 변경되었거나 처음 실행하는 경우 새로 생성
    else:
        st.info("새로운 민원편람 데이터를 AI가 읽고 학습합니다. 잠시만 기다려주세요...")
        # 1. PDF 로드 및 메타데이터 보강
        loader = PyPDFLoader(PDF_PATH)
        docs = loader.load()
        for doc in docs:
            page_num = doc.metadata.get('page', 0) + 1
            file_name = os.path.basename(PDF_PATH)
            doc.metadata['source_info'] = f"{file_name}의 {page_num}페이지"

        # 2. 텍스트 분할
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)

        # 3. 임베딩 및 벡터 DB 생성
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

        # 4. 벡터 DB와 해시 값 저장
        vectorstore.save_local(FAISS_INDEX_PATH)
        with open(HASH_FILE_PATH, 'w') as f:
            json.dump({'hash': current_pdf_hash}, f)
        st.success("AI가 민원편람 학습을 완료했습니다!")

    # 검색기(Retriever) 생성
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # AI에게 역할을 부여하는 프롬프트 (어르신 눈높이 맞춤)
    template = """
    당신은 전라남도 곡성군의 민원 업무를 안내하는 친절한 AI 상담원입니다.
    사용자의 질문에 대해 아래의 '참고 자료'를 바탕으로, 70대 이상 어르신께서 이해하기 쉽도록 답변해야 합니다.

    반드시 다음 규칙을 지켜주세요:
    1. 답변은 항상 "어르신, 문의하신 내용에 대해 알려드릴게요." 와 같은 친절한 인사로 시작하세요.
    2. 전문 용어는 최대한 피하고, 쉬운 단어로 풀어서 설명해주세요.
    3. 답변은 아래의 '답변 형식'에 맞춰, 각 항목을 굵은 글씨와 번호로 명확하게 구분해서 알려주세요.
    4. 각 항목의 내용 끝에는 반드시 정보의 출처를 정확히 명시해야 합니다. 예: [출처: minweonpyeonram-2025.pdf의 15페이지]
    5. 참고 자료에 내용이 없으면 "죄송합니다, 어르신. 해당 내용은 민원편람에서 찾을 수 없었습니다. 곡성군청 민원실(061-360-8262)로 문의하시면 더 정확한 안내를 받으실 수 있습니다." 라고 답변해주세요.

    ---
    [답변 형식]

    ## **어떤 민원인가요?**
    - [민원 업무에 대한 쉽고 간단한 설명] [출처]

    ### **1. 무엇을 준비해야 하나요? (구비서류)**
    - [필요한 서류 목록] [출처]

    ### **2. 어디로 가야 하나요? (신청장소/담당부서)**
    - [신청 장소와 담당 부서 이름] [출처]

    ### **3. 돈은 얼마나 드나요? (수수료)**
    - [수수료 금액, 면제 대상 등] [출처]

    ### **4. 처리하는 데 얼마나 걸리나요? (처리기간)**
    - [처리 기간] [출처]

    ---
    [참고 자료]
    {context}
    ---
    [사용자 질문]
    {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # LLM 모델 설정
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

    # RAG 체인 구성
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


