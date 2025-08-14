# rag_logic.py
import os
import re
import hashlib
from typing import List, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document


# ===== 해시 계산 =====
def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    """PDF 파일의 SHA-256 해시 계산"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ===== 디렉토리 보장 =====
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


# ===== PDF → FAISS 벡터스토어 준비 =====
def prepare_vectorstore(
    openai_api_key: str,
    pdf_paths: List[str],
    file_names: List[str],
    vector_dir: str = "faiss_minweonpyeonram_2025",
):
    """
    PDF를 임베딩하여 FAISS 인덱스 저장/로드
    - 최초 1회: 임베딩 후 save_local()
    - 이후: PDF 해시가 같으면 load_local()로 즉시 로딩
    """
    ensure_dir(vector_dir)
    hash_file = os.path.join(vector_dir, "hash.txt")

    # 모든 PDF 결합 해시
    combined_hash = "".join([sha256_file(p) for p in pdf_paths])
    combined_hash = hashlib.sha256(combined_hash.encode()).hexdigest()

    # 저장된 해시 읽기
    cached_hash = None
    if os.path.exists(hash_file):
        with open(hash_file, "r", encoding="utf-8") as hf:
            cached_hash = hf.read().strip()

    embeddings = OpenAIEmbeddings(
        openai_api_key=openai_api_key,
        model="text-embedding-3-small"
    )

    # 해시 동일 & 디렉토리 존재 → 로드
    if cached_hash == combined_hash and os.path.exists(vector_dir):
        print("[FAISS] 기존 인덱스 로드")
        return FAISS.load_local(vector_dir, embeddings, allow_dangerous_deserialization=True)

    # 해시 다름 → 새로 빌드
    print("[FAISS] 새 인덱스 생성 시작")
    all_docs: List[Document] = []
    for idx, pdf_path in enumerate(pdf_paths):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")

        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        fname = file_names[idx] if idx < len(file_names) else f"Document_{idx+1}"
        base_name = os.path.splitext(fname)[0]

        for doc_idx, doc in enumerate(docs):
            page_num = int(doc.metadata.get("page", doc_idx)) + 1
            doc.metadata["file_name"] = fname
            doc.metadata["page_number"] = page_num
            doc.metadata["source_info"] = f"{base_name} {page_num}p"

        all_docs.extend(docs)

    if not all_docs:
        raise ValueError("문서가 비어있거나 텍스트를 추출하지 못했습니다.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        length_function=len
    )
    chunks = splitter.split_documents(all_docs)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(vector_dir)

    with open(hash_file, "w", encoding="utf-8") as hf:
        hf.write(combined_hash)

    print("[FAISS] 인덱스 생성 완료 및 저장")
    return vectorstore


# ===== 검색기 생성 =====
def build_retriever(vectorstore: FAISS, k: int = 8):
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )


# ===== 별지/서식 추출 =====
FORM_KEYWORDS = ["별지", "서식", "신청서", "동의서", "양식", "정정신청서", "변경신고서", "등록신청서"]

def extract_annex_forms_from_docs(docs: List[Document]) -> List[str]:
    forms = []
    for doc in docs:
        for line in doc.page_content.splitlines():
            clean_line = line.strip()
            if any(k in clean_line for k in FORM_KEYWORDS):
                snippet = clean_line if len(clean_line) <= 60 else clean_line[:60] + "..."
                if snippet not in forms:
                    forms.append(snippet)
    return forms


# ===== 출처 포맷 =====
def format_source_line(doc: Document) -> str:
    fname = doc.metadata.get("file_name", "문서")
    page = doc.metadata.get("page_number", "?")
    preview = doc.page_content.strip().replace("\n", " ")
    if len(preview) > 80:
        preview = preview[:80] + "..."
    return f"{fname} p.{page} — \"{preview}\""


# ===== 컨텍스트 + 출처 + 서식 추출 =====
def make_context_and_sources(retriever, question: str) -> Tuple[str, List[str], List[str]]:
    docs: List[Document] = retriever.get_relevant_documents(question)
    if not docs:
        return "검색된 문서가 없습니다.", [], []

    sources = [format_source_line(d) for d in docs]
    annex_forms = extract_annex_forms_from_docs(docs)

    context_blocks = []
    for d in docs:
        src = d.metadata.get("source_info", format_source_line(d))
        content = d.page_content.strip()
        if len(content) > 1200:
            content = content[:1200] + "..."
        block = f"[출처: {src}]\n{content}"
        context_blocks.append(block)

    context_text = "\n\n".join(context_blocks)
    return context_text, sources, annex_forms


# ===== 스트리밍 LLM 생성 =====
def build_streaming_llm(model: str, openai_api_key: str, max_tokens: int = 800, temperature: float = 0):
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        openai_api_key=openai_api_key,
        max_tokens=max_tokens,
        streaming=True
    )


# ===== 프롬프트 생성 =====
def build_final_prompt(context: str, question: str, annex_forms: List[str]) -> str:
    forms_text = ""
    if annex_forms:
        forms_text = "\n\n[관련 별지/서식]\n" + "\n".join(f"- {f}" for f in annex_forms)

    return f"""
당신은 곡성군 민원 상담 전문가입니다.
아래 문맥을 토대로 질문에 대한 답변을 작성하세요.

**지침**
- 모든 제목은 본문과 동일 크기, 굵기만 적용
- 1) 민원업무명, 처리기간, 구비서류, 수수료 명시
- 2) 처리 절차 단계별 설명
- 3) 신청방법과 접수처, 담당부서 명시
- 4) 관련 법령/조례 정확히 인용
- 5) 가능한 경우 관련 별지/서식 안내
- 6) 불필요한 메타데이터나 처리시간, '근거 출처 모아보기'는 출력하지 않음

[문맥]
{context}
{forms_text}

[질문]
{question}

[답변]
"""



