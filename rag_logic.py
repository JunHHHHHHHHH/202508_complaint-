# rag_logic.py

import os
import re
from typing import List, Tuple, Optional, Dict, Any

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.schema import Document

from utils import file_md5, ensure_dir

FAISS_STORE_DIR = "./storage/faiss_minweonpyeonram_2025"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

def extract_issue_number(filename: str) -> str:
    pattern = r'제(\d+)호'
    match = re.search(pattern, filename)
    if match:
        return match.group(1)
    return "Unknown"

def _build_source_info(file_name: str, page_num: int) -> str:
    base_filename = os.path.splitext(file_name)[0]
    return f"{base_filename}의 {page_num}p"

def _load_pdf_with_metadata(pdf_path: str, display_name: str) -> List[Document]:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    for i, doc in enumerate(docs):
        doc.metadata['file_name'] = display_name
        doc.metadata['document_name'] = display_name
        original_page = doc.metadata.get('page', i)
        page_num = int(original_page) + 1
        doc.metadata['page_number'] = page_num
        doc.metadata['source_info'] = _build_source_info(display_name, page_num)
    return docs

def _split_docs(all_docs: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        length_function=len
    )
    splits = text_splitter.split_documents(all_docs)
    for split in splits:
        if 'source_info' not in split.metadata:
            file_name = split.metadata.get('file_name', 'Unknown')
            page_num = split.metadata.get('page_number', 'Unknown')
            split.metadata['source_info'] = _build_source_info(file_name, page_num)
        preview = split.page_content[:100] + ("..." if len(split.page_content) > 100 else "")
        split.metadata['content_preview'] = preview
    return splits

def _save_faiss(vectorstore: FAISS, store_dir: str):
    ensure_dir(store_dir)
    vectorstore.save_local(store_dir)

def _load_faiss(store_dir: str, embeddings: OpenAIEmbeddings) -> Optional[FAISS]:
    if not os.path.isdir(store_dir):
        return None
    try:
        vs = FAISS.load_local(store_dir, embeddings=embeddings, allow_dangerous_deserialization=True)
        return vs
    except Exception:
        return None

def _faiss_stamp_path(store_dir: str) -> str:
    return os.path.join(store_dir, "SOURCE_HASH.txt")

def _read_stamp(store_dir: str) -> Optional[str]:
    stamp_path = _faiss_stamp_path(store_dir)
    if not os.path.exists(stamp_path):
        return None
    try:
        with open(stamp_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return None

def _write_stamp(store_dir: str, value: str):
    ensure_dir(store_dir)
    with open(_faiss_stamp_path(store_dir), "w", encoding="utf-8") as f:
        f.write(value)

def initialize_vectorstore(openai_api_key: str, pdf_path: str, display_name: str) -> Tuple[FAISS, OpenAIEmbeddings]:
    if not openai_api_key or not openai_api_key.startswith("sk-"):
        raise ValueError("유효한 OpenAI API 키를 입력해주세요.")

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model=EMBED_MODEL)

    # PDF 해시 계산
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
    pdf_hash = file_md5(pdf_path)

    # 저장된 인덱스가 있고, 스탬프 해시가 같으면 로드
    existing = _load_faiss(FAISS_STORE_DIR, embeddings)
    stamp = _read_stamp(FAISS_STORE_DIR)
    if existing is not None and stamp == pdf_hash:
        return existing, embeddings

    # 새로 구축
    all_docs = _load_pdf_with_metadata(pdf_path, display_name)
    if not all_docs:
        raise ValueError("PDF에서 텍스트를 추출할 수 없습니다.")
    total_text_len = sum(len(d.page_content) for d in all_docs)
    if total_text_len < 100:
        raise ValueError("문서 내용이 너무 짧습니다. 스캔된 이미지 PDF일 수 있습니다.")

    splits = _split_docs(all_docs)
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    _save_faiss(vectorstore, FAISS_STORE_DIR)
    _write_stamp(FAISS_STORE_DIR, pdf_hash)
    return vectorstore, embeddings

def make_retriever(vectorstore: FAISS):
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

def build_prompt() -> ChatPromptTemplate:
    template = """당신은 곡성군 민원 상담 전문가입니다. 70대 이상 어르신도 이해하기 쉽도록 쉬운 말로, 단계별로, 자세하게 안내하세요.
민원명-대상-신청방법-구비서류-처리기간-수수료-절차-법적근거-접수처-서식-유의사항-출처 순으로 답변합니다.

중요:
- 법률 명칭은 괄호로 간단히만 덧붙이고, 먼저 쉬운 설명으로 풀어주세요.
- 각 항목을 짧고 명확한 문장으로, 목록 형태로 설명하세요.
- 반드시 “출처: 파일명의 페이지”를 답변 맨 아래에 모아서 표시하세요.
- 아래 문맥에는 각 단락 앞에 [출처: 파일명의 Xp]가 붙습니다. 해당 출처를 답변에 반영하세요.

[문맥]
{context}

[질문]
{question}

[답변]
"""
    return ChatPromptTemplate.from_template(template)

def _format_docs_for_context(docs: List[Document]) -> str:
    formatted = []
    for doc in docs:
        source = doc.metadata.get('source_info', 'Unknown')
        content = doc.page_content
        formatted.append(f"[출처: {source}]\n{content}")
    return "\n\n".join(formatted)

def _collect_sources(docs: List[Document]) -> List[str]:
    seen = set()
    ordered = []
    for d in docs:
        s = d.metadata.get("source_info")
        if s and s not in seen:
            seen.add(s)
            ordered.append(s)
    return ordered

def initialize_rag_chain(openai_api_key: str, pdf_path: str, display_name: str = "minweonpyeonram-2025.pdf"):
    vectorstore, embeddings = initialize_vectorstore(openai_api_key, pdf_path, display_name)
    retriever = make_retriever(vectorstore)
    prompt = build_prompt()
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0, openai_api_key=openai_api_key, max_tokens=1000, timeout=60)

    def format_docs(docs: List[Document]) -> str:
        return _format_docs_for_context(docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain, retriever

def get_answer(chain, retriever, question: str, openai_api_key: str) -> str:
    # 검색
    docs = retriever.get_relevant_documents(question)
    if not docs:
        return "해당 문서에서 관련 정보를 찾을 수 없습니다."

    # 컨텍스트 생성
    context = _format_docs_for_context(docs)

    # 프롬프트
    prompt = build_prompt()
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0, openai_api_key=openai_api_key, max_tokens=1000, timeout=60)
    final_prompt = prompt.format(context=context, question=question)
    resp = llm.invoke(final_prompt)

    # 하단 출처 단 한번 더 수집하여 붙이기
    sources = _collect_sources(docs)
    if sources:
        src_text = "\n".join(f"- {s}" for s in sources)
        # 모델 답변 뒤에 “출처” 섹션을 추가(중복 방지)
        content = resp.content.strip()
        if "출처" not in content:
            content += f"\n\n출처\n{src_text}"
        else:
            content += f"\n{src_text}"
        return content
    return resp.content.strip()

