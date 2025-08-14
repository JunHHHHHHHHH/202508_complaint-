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


# ===== 유틸 =====
def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


# ===== PDF → 벡터 인덱스 준비(저장/로드) =====
def prepare_vectorstore(
    openai_api_key: str,
    pdf_paths: List[str],
    file_names: List[str],
    vector_dir: str = "faiss_minweonpyeonram_2025",
):
    """
    - pdf_paths의 단일 PDF(또는 여러 PDF)를 임베딩 → FAISS 생성 후 vector_dir에 저장
    - 다음 실행 시: PDF 해시 동일하면 load_local로 즉시 로드
    """
    ensure_dir(vector_dir)
    hash_file = os.path.join(vector_dir, "hash.txt")

    # PDF 결합 해시(복수 PDF 지원)
    combined_hash = ""
    for p in pdf_paths:
        combined_hash += sha256_file(p)
    combined_hash = hashlib.sha256(combined_hash.encode()).hexdigest()

    # 저장된 해시와 비교
    cached_hash = None
    if os.path.exists(hash_file):
        try:
            with open(hash_file, "r", encoding="utf-8") as hf:
                cached_hash = hf.read().strip()
        except:
            cached_hash = None

    # Embeddings 준비
    embeddings = OpenAIEmbeddings(
        openai_api_key=openai_api_key,
        model="text-embedding-3-small"
    )

    # 해시 동일 → 로드
    if cached_hash and cached_hash == combined_hash and os.path.exists(vector_dir):
        # 주의: load_local 시 Embeddings 인스턴스 필요
        vectorstore = FAISS.load_local(
            vector_dir, embeddings, allow_dangerous_deserialization=True
        )
        return vectorstore

    # 해시 다름 → 새로 빌드
    all_docs: List[Document] = []
    for i, pdf_path in enumerate(pdf_paths):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")

        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        # 메타데이터 채우기
        fname = file_names[i] if i < len(file_names) else f"Document_{i+1}"
        for doc_idx, doc in enumerate(docs):
            original_page = doc.metadata.get("page", doc_idx)
            page_num = int(original_page) + 1
            doc.metadata["file_name"] = fname
            doc.metadata["page_number"] = page_num
            base_name = os.path.splitext(fname)[0]
            doc.metadata["source_info"] = f"{base_name} {page_num}p"
        all_docs.extend(docs)

    if not all_docs:
        raise ValueError("문서가 비어있거나 텍스트를 추출하지 못했습니다.")

    # 분할
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        length_function=len
    )
    chunks = splitter.split_documents(all_docs)

    # 벡터스토어 생성 및 저장
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(vector_dir)

    # 해시 저장
    with open(hash_file, "w", encoding="utf-8") as hf:
        hf.write(combined_hash)

    return vectorstore


def build_retriever(vectorstore: FAISS, k: int = 8):
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )


# ===== 출처 강화 포맷 / 별지서식 추출 / 컨텍스트 생성 =====
FORM_KEYWORDS = ["별지", "서식", "신청서", "동의서", "양식", "정정신청서", "변경신고서", "등록신청서"]

def format_source_line(doc: Document) -> str:
    """
    예: "곡성군 민원편람 2025 p.12 — '여권 발급 신청서 제출 ...'"
    """
    fname = doc.metadata.get("file_name", "문서")
    page = doc.metadata.get("page_number", "?")
    preview = doc.page_content.strip().replace("\n", " ")
    if len(preview) > 80:
        preview = preview[:80] + "..."
    return f"{fname} p.{page} — \"{preview}\""

def extract_annex_forms_from_docs(docs: List[Document]) -> List[str]:
    forms = []
    for d in docs:
        text = d.page_content
        # 줄 단위로 스캔하여 '별지'나 '서식' 키워드 인접 문구를 최대한 간명히 추출
        for line in text.splitlines():
            line_clean = line.strip()
            if any(k in line_clean for k in FORM_KEYWORDS):
                # 과도하게 긴 라인은 잘라서 간결히
                snippet = line_clean
                if len(snippet) > 60:
                    snippet = snippet[:60] + "..."
                if snippet not in forms:
                    forms.append(snippet)
    return forms

def make_context_and_sources(retriever, question: str) -> Tuple[str, List[str], List[str]]:
    docs: List[Document] = retriever.get_relevant_documents(question)
    if not docs:
        return "검색된 문서가 없습니다.", [], []

    # 상위 문서에서 출처/서식 추출
    sources = [format_source_line(d) for d in docs]
    annex_forms = extract_annex_forms_from_docs(docs)

    # 컨텍스트(출처 포함) 생성. 각 청크는 1,200자 제한으로 과도한 길이 방지
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


# ===== 스트리밍 LLM & 프롬프트 =====
def build_streaming_llm(model: str, openai_api_key: str, max_tokens: int = 800, temperature: float = 0):
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        openai_api_key=openai_api_key,
        max_tokens=max_tokens,
        streaming=True
    )

def build_final_prompt(context: str, question: str, annex_forms: List[str]) -> str:
    forms_text = ""
    if annex_forms:
        forms_text = "\n\n[관련 별지/서식]\n" + "\n".join(f"- {f}" for f in annex_forms)

    return f"""
당신은 곡성군 민원 상담 전문가입니다. 곡성군 민원편람을 바탕으로 정확하고 친절하게 답변하세요.

답변 지침:
1) 민원업무명, 처리기간, 구비서류, 수수료를 명확히 제시
2) 처리 절차를 단계별로 설명
3) 신청방법(방문/온라인/전화 등)과 접수처, 담당부서 명시
4) 관련 법령/조례/근거를 정확히 인용
5) 답변 말미에 반드시 [출처 표기]를 유지
6) 가능한 경우 관련 별지/서식명을 함께 안내

[문맥]
{context}
{forms_text}

[질문]
{question}

[답변]
"""

