```python
import os
import re
from typing import List, Dict, Tuple

import faiss
import numpy as np
import streamlit as st
import fitz  # PyMuPDF

from sentence_transformers import SentenceTransformer
from groq import Groq


# ============================================================
# CONFIGURATION
# ============================================================

APP_NAME = "BidReady AI"

GROQ_MODEL = "openai/gpt-oss-20b"

# Lightweight and strong general-purpose embedding model.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 6

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="BidReady AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 3rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 1.15rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }

    .decision-card {
        padding: 25px;
        border-radius: 18px;
        text-align: center;
        margin: 10px 0 25px 0;
    }

    .score {
        font-size: 3.5rem;
        font-weight: 800;
    }

    .decision {
        font-size: 1.35rem;
        font-weight: 700;
    }

    .evidence-card {
        padding: 15px;
        border-radius: 12px;
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        margin-bottom: 12px;
    }

    .small-label {
        color: #64748B;
        font-size: 0.85rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "documents" not in st.session_state:
    st.session_state.documents = []

if "index" not in st.session_state:
    st.session_state.index = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None

if "documents_ready" not in st.session_state:
    st.session_state.documents_ready = False


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


def get_groq_client():
    """
    Get Groq API client from Streamlit secrets.

    Streamlit Cloud:
        GROQ_API_KEY = "..."

    Local environment:
        GROQ_API_KEY environment variable
    """

    api_key = None

    # Streamlit secrets
    try:
        api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        pass

    # Environment variable fallback
    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY was not found. "
            "Add it to Streamlit Secrets."
        )

    return Groq(api_key=api_key)


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf_pages(
    uploaded_file,
    document_type: str,
) -> List[Dict]:

    pdf_bytes = uploaded_file.getvalue()

    try:
        document = fitz.open(
            stream=pdf_bytes,
            filetype="pdf",
        )
    except Exception as exc:
        raise ValueError(
            f"Could not open {uploaded_file.name}: {exc}"
        )

    pages = []

    for page_number, page in enumerate(
        document,
        start=1,
    ):

        text = page.get_text("text").strip()

        if not text:
            continue

        pages.append(
            {
                "document": uploaded_file.name,
                "document_type": document_type,
                "page": page_number,
                "text": text,
            }
        )

    document.close()

    return pages


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text: str) -> str:

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


# ============================================================
# CHUNKING
# ============================================================

def chunk_page(
    page: Dict,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[Dict]:

    text = clean_text(page["text"])

    if len(text) <= chunk_size:

        return [
            {
                **page,
                "chunk_id": (
                    f"{page['document']}"
                    f"-p{page['page']}-c1"
                ),
                "text": text,
            }
        ]

    chunks = []

    start = 0
    chunk_number = 1

    while start < len(text):

        end = min(
            start + chunk_size,
            len(text),
        )

        chunk_text = text[start:end].strip()

        if chunk_text:

            chunks.append(
                {
                    **page,
                    "chunk_id": (
                        f"{page['document']}"
                        f"-p{page['page']}"
                        f"-c{chunk_number}"
                    ),
                    "text": chunk_text,
                }
            )

        if end >= len(text):
            break

        start = max(
            end - overlap,
            start + 1,
        )

        chunk_number += 1

    return chunks


def create_chunks(
    pages: List[Dict],
) -> List[Dict]:

    all_chunks = []

    for page in pages:

        all_chunks.extend(
            chunk_page(page)
        )

    return all_chunks


# ============================================================
# BUILD FAISS INDEX
# ============================================================

def build_faiss_index(
    chunks: List[Dict],
    embedding_model,
):

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    embeddings = embeddings.astype(
        "float32"
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(embeddings)

    return index


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(
    query: str,
    index,
    chunks: List[Dict],
    embedding_model,
    top_k: int = TOP_K,
) -> List[Dict]:

    if index is None or not chunks:
        return []

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    query_embedding = query_embedding.astype(
        "float32"
    )

    scores, indices = index.search(
        query_embedding,
        min(top_k, len(chunks)),
    )

    results = []

    for score, index_number in zip(
        scores[0],
        indices[0],
    ):

        if index_number < 0:
            continue

        chunk = dict(
            chunks[index_number]
        )

        chunk["similarity"] = float(score)

        results.append(chunk)

    return results


# ============================================================
# FORMAT RETRIEVED EVIDENCE
# ============================================================

def format_context(
    results: List[Dict],
) -> str:

    if not results:
        return "No relevant evidence was retrieved."

    sections = []

    for number, result in enumerate(
        results,
        start=1,
    ):

        sections.append(
            f"""
[EVIDENCE {number}]
Document: {result['document']}
Document Type: {result['document_type']}
Page: {result['page']}
Similarity: {result['similarity']:.3f}

Content:
{result['text']}
"""
        )

    return "\n".join(sections)


# ============================================================
# GROQ ANALYSIS
# ============================================================

def analyze_with_groq(
    question: str,
    company_profile: str,
    retrieved_results: List[Dict],
) -> str:

    client = get_groq_client()

    context = format_context(
        retrieved_results
    )

    system_prompt = """
You are BidReady AI, an expert tender-readiness assistant.

Your job is to help a company determine whether a tender appears
suitable for them.

You MUST follow these rules:

1. Use the retrieved evidence as your primary source.
2. Do not invent tender requirements.
3. Do not invent company capabilities.
4. If the evidence is insufficient, say NEEDS REVIEW.
5. Distinguish between:
   - Match
   - Partial Match
   - Gap
   - Needs Review
6. Pay special attention to mandatory requirements.
7. Cite evidence using document name and page number.
8. Never fabricate a page number.
9. Be conservative about bid recommendations.
10. Explain your reasoning clearly.

For suitability questions, consider:

- Eligibility
- Technical capability
- Relevant experience
- Certifications
- Registrations
- Financial requirements
- Documentation
- Geographic requirements
- Deadlines
- Mandatory conditions

If a mandatory requirement appears to be missing,
highlight it prominently.

A high number of matches does NOT automatically mean the tender
is suitable if an important mandatory requirement is missing.

Possible recommendations:

BID
BID WITH CONDITIONS
NO-BID
NEEDS REVIEW
"""

    user_prompt = f"""
COMPANY PROFILE
================

{company_profile}


USER QUESTION
================

{question}


RETRIEVED TENDER / COMPANY EVIDENCE
=====================================

{context}


TASK
================

Answer the user's question using the evidence above.

When appropriate, structure your response as:

1. Direct Answer
2. Readiness Assessment
3. Matching Requirements
4. Gaps / Risks
5. Evidence
6. Recommended Next Steps

If you cannot confidently determine suitability from the available
evidence, say NEEDS REVIEW rather than guessing.
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0.1,
        max_tokens=2500,
    )

    return response.choices[0].message.content


# ============================================================
# DOCUMENT INGESTION
# ============================================================

def process_documents(
    tender_file,
    company_file,
):

    all_pages = []

    tender_pages = extract_pdf_pages(
        tender_file,
        "Tender",
    )

    company_pages = extract_pdf_pages(
        company_file,
        "Company Profile",
    )

    all_pages.extend(
        tender_pages
    )

    all_pages.extend(
        company_pages
    )

    if not all_pages:
        raise ValueError(
            "No readable text was found in the uploaded PDFs."
        )

    chunks = create_chunks(
        all_pages
    )

    embedding_model = load_embedding_model()

    index = build_faiss_index(
        chunks,
        embedding_model,
    )

    return (
        all_pages,
        chunks,
        index,
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🎯 BidReady AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        Know Before You Bid — RAG-powered tender intelligence
        for businesses.
    </div>
    """,
    unsafe_allow_html=True,
)

st.write(
    "Upload your company profile and tender, then ask questions "
    "about eligibility, requirements, risks, and bid suitability."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Configuration")

    st.info(
        "This application uses:\n\n"
        "• PyMuPDF for PDF extraction\n"
        "• Sentence Transformers for embeddings\n"
        "• FAISS for vector retrieval\n"
        "• Groq for LLM reasoning"
    )

    st.divider()

    st.subheader("Required API Key")

    st.code(
        "GROQ_API_KEY",
        language="text",
    )

    st.caption(
        "Add your Groq API key through Streamlit "
        "Cloud Secrets."
    )

    st.divider()

    if st.session_state.documents_ready:

        st.success(
            "Knowledge base ready"
        )

        st.metric(
            "Indexed chunks",
            len(
                st.session_state.chunks
            ),
        )

    else:

        st.warning(
            "Upload both PDFs to build the knowledge base."
        )


# ============================================================
# DOCUMENT UPLOAD
# ============================================================

st.header("📄 1. Upload Documents")

col1, col2 = st.columns(2)

with col1:

    tender_file = st.file_uploader(
        "Tender / RFP PDF",
        type=["pdf"],
        key="tender_pdf",
        help=(
            "Upload the tender you want to evaluate."
        ),
    )

with col2:

    company_file = st.file_uploader(
        "Company Profile PDF",
        type=["pdf"],
        key="company_pdf",
        help=(
            "Upload your company's profile, "
            "capabilities, experience, certifications, etc."
        ),
    )


# ============================================================
# BUILD KNOWLEDGE BASE
# ============================================================

if tender_file and company_file:

    if st.button(
        "🔎 Build BidReady Knowledge Base",
        type="primary",
        use_container_width=True,
    ):

        with st.spinner(
            "Extracting PDFs, creating chunks, "
            "generating embeddings, and building FAISS index..."
        ):

            try:

                pages, chunks, index = process_documents(
                    tender_file,
                    company_file,
                )

                st.session_state.documents = pages
                st.session_state.chunks = chunks
                st.session_state.index = index
                st.session_state.embedding_model = (
                    load_embedding_model()
                )
                st.session_state.documents_ready = True

                # Clear previous conversation when
                # new documents are uploaded.
                st.session_state.messages = []

                st.success(
                    f"Knowledge base created successfully. "
                    f"{len(chunks)} chunks indexed."
                )

            except Exception as exc:

                st.error(
                    f"Could not build knowledge base: {exc}"
                )


# ============================================================
# DOCUMENT SUMMARY
# ============================================================

if st.session_state.documents_ready:

    st.header("📚 Knowledge Base")

    tender_pages_count = sum(
        1
        for item in st.session_state.documents
        if item["document_type"] == "Tender"
    )

    company_pages_count = sum(
        1
        for item in st.session_state.documents
        if item["document_type"] == "Company Profile"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Tender pages",
        tender_pages_count,
    )

    col2.metric(
        "Company profile pages",
        company_pages_count,
    )

    col3.metric(
        "Indexed chunks",
        len(st.session_state.chunks),
    )

    with st.expander(
        "🔍 View indexed document information"
    ):

        st.dataframe(
            [
                {
                    "Document": item["document"],
                    "Type": item["document_type"],
                    "Page": item["page"],
                }
                for item in st.session_state.documents
            ],
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# CHAT / QUESTIONS
# ============================================================

if st.session_state.documents_ready:

    st.divider()

    st.header("💬 2. Ask BidReady AI")

    st.caption(
        "Ask whether the tender is suitable, what requirements "
        "you meet, what gaps exist, or what you should do before bidding."
    )

    example_questions = [
        "Is this tender suitable for my company?",
        "What mandatory requirements does my company fail to meet?",
        "Which tender requirements does my company satisfy?",
        "What are the biggest risks if we bid?",
        "Do we have enough relevant experience?",
        "What certifications are required?",
        "What documents are missing?",
        "Should we BID, BID WITH CONDITIONS, or NO-BID?",
    ]

    selected_question = st.selectbox(
        "Example questions",
        ["Choose a question..."]
        + example_questions,
    )

    question = st.chat_input(
        "Ask BidReady AI a question..."
    )

    if (
        not question
        and selected_question != "Choose a question..."
    ):
        question = selected_question

    if question:

        # Display user question
        st.chat_message(
            "user"
        ).write(question)

        with st.spinner(
            "Retrieving evidence and analyzing suitability..."
        ):

            try:

                embedding_model = (
                    st.session_state.embedding_model
                )

                results = retrieve(
                    query=question,
                    index=st.session_state.index,
                    chunks=st.session_state.chunks,
                    embedding_model=embedding_model,
                    top_k=TOP_K,
                )

                # Build company profile from all company chunks
                company_chunks = [
                    chunk
                    for chunk in st.session_state.chunks
                    if chunk["document_type"]
                    == "Company Profile"
                ]

                company_profile = "\n\n".join(
                    chunk["text"]
                    for chunk in company_chunks
                )

                answer = analyze_with_groq(
                    question=question,
                    company_profile=company_profile,
                    retrieved_results=results,
                )

                st.session_state.messages.append(
                    {
                        "question": question,
                        "answer": answer,
                    }
                )

                st.chat_message(
                    "assistant"
                ).markdown(answer)

                # ----------------------------------------
                # RETRIEVED EVIDENCE
                # ----------------------------------------

                with st.expander(
                    f"🔎 Retrieved Evidence ({len(results)} chunks)"
                ):

                    for number, result in enumerate(
                        results,
                        start=1,
                    ):

                        st.markdown(
                            f"""
                            <div class="evidence-card">

                            <strong>
                            Evidence {number}
                            </strong>

                            <br>

                            <span class="small-label">
                            {result['document']}
                            • Page {result['page']}
                            • Similarity {result['similarity']:.3f}
                            </span>

                            <br><br>

                            {result['text']}

                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            except Exception as exc:

                st.error(
                    f"Analysis failed: {exc}"
                )

else:

    st.info(
        "👆 Upload both a tender PDF and a company profile PDF, "
        "then build the knowledge base."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "BidReady AI is an AI-assisted decision-support tool. "
    "Always verify important requirements against the original "
    "tender documentation before making a final bid decision."
)
```
