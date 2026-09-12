import os
import re
from typing import Dict, List

import faiss
import fitz
import numpy as np
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

APP_NAME = "BidReady AI"

GROQ_MODEL = "openai/gpt-oss-20b"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K_TENDER = 5
TOP_K_COMPANY = 5

SIMILARITY_THRESHOLD = 0.25

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title=APP_NAME,
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
        margin: 15px 0 25px 0;
        border: 1px solid #E2E8F0;
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

    .source-label {
        color: #475569;
        font-size: 0.85rem;
        font-weight: 600;
    }

    .warning-card {
        padding: 15px;
        border-radius: 10px;
        background: #FFF7ED;
        border-left: 5px solid #EA580C;
        margin: 10px 0;
    }

    .success-card {
        padding: 15px;
        border-radius: 10px;
        background: #F0FDF4;
        border-left: 5px solid #16A34A;
        margin: 10px 0;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "messages": [],
    "tender_pages": [],
    "company_pages": [],
    "tender_chunks": [],
    "company_chunks": [],
    "tender_index": None,
    "company_index": None,
    "embedding_model": None,
    "documents_ready": False,
    "tender_name": "",
    "company_name": "",
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


# ============================================================
# GROQ CLIENT
# ============================================================

def get_groq_client() -> Groq:
    api_key = None

    try:
        api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        pass

    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not configured. "
            "Add GROQ_API_KEY to Streamlit Secrets."
        )

    return Groq(api_key=api_key)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf_pages(
    uploaded_file,
    document_type: str,
) -> List[Dict]:

    pdf_bytes = uploaded_file.getvalue()

    if not pdf_bytes:
        raise ValueError(
            f"{uploaded_file.name} is empty."
        )

    try:
        document = fitz.open(
            stream=pdf_bytes,
            filetype="pdf",
        )
    except Exception as exc:
        raise ValueError(
            f"Could not open {uploaded_file.name}: {exc}"
        ) from exc

    pages = []

    try:
        for page_number, page in enumerate(
            document,
            start=1,
        ):
            text = page.get_text("text")
            text = clean_text(text)

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
    finally:
        document.close()

    return pages


# ============================================================
# CHUNKING
# ============================================================

def chunk_page(
    page: Dict,
) -> List[Dict]:

    text = page["text"]

    if len(text) <= CHUNK_SIZE:
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
            start + CHUNK_SIZE,
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
            end - CHUNK_OVERLAP,
            start + 1,
        )

        chunk_number += 1

    return chunks


def create_chunks(
    pages: List[Dict],
) -> List[Dict]:

    chunks = []

    for page in pages:
        chunks.extend(
            chunk_page(page)
        )

    return chunks


# ============================================================
# FAISS INDEX
# ============================================================

def build_faiss_index(
    chunks: List[Dict],
    embedding_model,
):

    if not chunks:
        raise ValueError(
            "No chunks available to index."
        )

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

    embeddings = np.asarray(
        embeddings,
        dtype="float32",
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
    top_k: int,
) -> List[Dict]:

    if (
        index is None
        or not chunks
        or not query.strip()
    ):
        return []

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32",
    )

    k = min(
        top_k,
        len(chunks),
    )

    scores, indices = index.search(
        query_embedding,
        k,
    )

    results = []

    for score, index_number in zip(
        scores[0],
        indices[0],
    ):

        if index_number < 0:
            continue

        similarity = float(score)

        if similarity < SIMILARITY_THRESHOLD:
            continue

        chunk = dict(
            chunks[index_number]
        )

        chunk["similarity"] = similarity

        results.append(chunk)

    return results


# ============================================================
# FORMAT RAG CONTEXT
# ============================================================

def format_context(
    tender_results: List[Dict],
    company_results: List[Dict],
) -> str:

    sections = []

    if tender_results:

        sections.append(
            "========== TENDER EVIDENCE =========="
        )

        for number, result in enumerate(
            tender_results,
            start=1,
        ):

            sections.append(
                f"""
[TENDER EVIDENCE {number}]
Document: {result["document"]}
Page: {result["page"]}
Similarity: {result["similarity"]:.3f}

Content:
{result["text"]}
"""
            )

    if company_results:

        sections.append(
            "========== COMPANY EVIDENCE =========="
        )

        for number, result in enumerate(
            company_results,
            start=1,
        ):

            sections.append(
                f"""
[COMPANY EVIDENCE {number}]
Document: {result["document"]}
Page: {result["page"]}
Similarity: {result["similarity"]:.3f}

Content:
{result["text"]}
"""
            )

    if not sections:
        return (
            "No sufficiently relevant evidence "
            "was retrieved."
        )

    return "\n".join(sections)


# ============================================================
# GROQ ANALYSIS
# ============================================================

def analyze_with_groq(
    question: str,
    tender_results: List[Dict],
    company_results: List[Dict],
) -> str:

    client = get_groq_client()

    context = format_context(
        tender_results,
        company_results,
    )

    system_prompt = """
You are BidReady AI, an evidence-grounded tender
readiness assistant.

Your purpose is to help a business assess whether
a tender appears suitable for its company.

IMPORTANT EVIDENCE RULES:

1. Use only the retrieved tender and company evidence.
2. Do not invent facts.
3. Do not assume that missing evidence means the company fails.
4. When evidence is insufficient, say NEEDS REVIEW.
5. Never fabricate page numbers.
6. Every important factual claim should identify its source.
7. Distinguish tender requirements from company capabilities.
8. Pay special attention to mandatory requirements.
9. Be conservative with bid recommendations.

For each important requirement, classify it as:

MATCH
PARTIAL MATCH
GAP
NEEDS REVIEW

Possible overall recommendations:

BID
BID WITH CONDITIONS
NO-BID
NEEDS REVIEW

A mandatory requirement that appears to be unmet
is more important than a large number of minor matches.

Do not provide legal or official procurement advice.
"""

    user_prompt = f"""
# USER QUESTION

{question}

# RETRIEVED EVIDENCE

{context}

# TASK

Answer the user's question using the retrieved evidence.

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

Use this structure when appropriate:

## Direct Answer

Give the answer first.

## Suitability Assessment

Explain whether the tender appears suitable.

## Requirements We Match

List important matches.

## Gaps and Risks

List important gaps, partial matches, and risks.

## Evidence

Cite document name and page number.

## Recommendation

Use one of:

BID
BID WITH CONDITIONS
NO-BID
NEEDS REVIEW

## Next Steps

Give practical actions.

If the evidence is insufficient, do not guess.
Use NEEDS REVIEW.
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
        max_completion_tokens=3000,
        include_reasoning=False,
    )

    answer = response.choices[0].message.content

    if not answer:
        raise RuntimeError(
            "Groq returned an empty response."
        )

    return answer


# ============================================================
# BUILD KNOWLEDGE BASE
# ============================================================

def process_documents(
    tender_file,
    company_file,
):

    tender_pages = extract_pdf_pages(
        tender_file,
        "Tender",
    )

    company_pages = extract_pdf_pages(
        company_file,
        "Company Profile",
    )

    if not tender_pages:
        raise ValueError(
            "No readable text was found in the tender PDF. "
            "Please upload a searchable PDF."
        )

    if not company_pages:
        raise ValueError(
            "No readable text was found in the company profile PDF. "
            "Please upload a searchable PDF."
        )

    tender_chunks = create_chunks(
        tender_pages
    )

    company_chunks = create_chunks(
        company_pages
    )

    embedding_model = load_embedding_model()

    tender_index = build_faiss_index(
        tender_chunks,
        embedding_model,
    )

    company_index = build_faiss_index(
        company_chunks,
        embedding_model,
    )

    return (
        tender_pages,
        company_pages,
        tender_chunks,
        company_chunks,
        tender_index,
        company_index,
        embedding_model,
    )


# ============================================================
# DISPLAY EVIDENCE
# ============================================================

def display_evidence(
    title: str,
    results: List[Dict],
):

    with st.expander(
        f"{title} ({len(results)} chunks)"
    ):

        if not results:

            st.warning(
                "No sufficiently relevant evidence was retrieved."
            )

            return

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

                <span class="source-label">
                {result["document"]}
                • Page {result["page"]}
                • Similarity {result["similarity"]:.3f}
                </span>

                <br><br>

                {result["text"]}

                </div>
                """,
                unsafe_allow_html=True,
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
    Know Before You Bid — evidence-grounded RAG
    for tender suitability analysis.
    </div>
    """,
    unsafe_allow_html=True,
)

st.write(
    "Upload a tender and your company profile. "
    "BidReady AI retrieves relevant evidence from both "
    "documents and uses Groq to assess suitability."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ System")

    st.markdown(
        f"""
        **LLM**

        `{GROQ_MODEL}`

        **Embeddings**

        `{EMBEDDING_MODEL}`

        **Vector Database**

        FAISS

        **PDF Extraction**

        PyMuPDF
        """
    )

    st.divider()

    st.subheader(
        "API Configuration"
    )

    st.code(
        "GROQ_API_KEY",
        language="text",
    )

    st.caption(
        "Configure the API key through "
        "Streamlit Secrets. Never commit "
        "the key to GitHub."
    )

    st.divider()

    if st.session_state.documents_ready:

        st.success(
            "Knowledge base ready"
        )

        st.metric(
            "Tender chunks",
            len(
                st.session_state.tender_chunks
            ),
        )

        st.metric(
            "Company chunks",
            len(
                st.session_state.company_chunks
            ),
        )

    else:

        st.warning(
            "Upload both PDFs and build the knowledge base."
        )


# ============================================================
# DOCUMENT UPLOAD
# ============================================================

st.header(
    "📄 1. Upload Documents"
)

col1, col2 = st.columns(2)

with col1:

    tender_file = st.file_uploader(
        "Tender / RFP PDF",
        type=["pdf"],
        key="tender_pdf",
        help="Upload the tender you want to evaluate.",
    )

with col2:

    company_file = st.file_uploader(
        "Company Profile PDF",
        type=["pdf"],
        key="company_pdf",
        help=(
            "Upload your company profile, "
            "experience, certifications, capabilities, "
            "financial information, etc."
        ),
    )


# ============================================================
# BUILD KNOWLEDGE BASE
# ============================================================

if tender_file and company_file:

    st.divider()

    if st.button(
        "🔎 Build BidReady Knowledge Base",
        type="primary",
        use_container_width=True,
    ):

        with st.spinner(
            "Extracting documents, creating chunks, "
            "generating embeddings, and building FAISS indexes..."
        ):

            try:

                (
                    tender_pages,
                    company_pages,
                    tender_chunks,
                    company_chunks,
                    tender_index,
                    company_index,
                    embedding_model,
                ) = process_documents(
                    tender_file,
                    company_file,
                )

                st.session_state.tender_pages = (
                    tender_pages
                )

                st.session_state.company_pages = (
                    company_pages
                )

                st.session_state.tender_chunks = (
                    tender_chunks
                )

                st.session_state.company_chunks = (
                    company_chunks
                )

                st.session_state.tender_index = (
                    tender_index
                )

                st.session_state.company_index = (
                    company_index
                )

                st.session_state.embedding_model = (
                    embedding_model
                )

                st.session_state.documents_ready = True

                st.session_state.messages = []

                st.session_state.tender_name = (
                    tender_file.name
                )

                st.session_state.company_name = (
                    company_file.name
                )

                st.success(
                    "BidReady knowledge base created successfully."
                )

            except Exception as exc:

                st.error(
                    f"Could not build knowledge base: {exc}"
                )


# ============================================================
# KNOWLEDGE BASE SUMMARY
# ============================================================

if st.session_state.documents_ready:

    st.header(
        "📚 Knowledge Base"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Tender Pages",
        len(
            st.session_state.tender_pages
        ),
    )

    col2.metric(
        "Company Pages",
        len(
            st.session_state.company_pages
        ),
    )

    col3.metric(
        "Tender Chunks",
        len(
            st.session_state.tender_chunks
        ),
    )

    col4.metric(
        "Company Chunks",
        len(
            st.session_state.company_chunks
        ),
    )

    with st.expander(
        "🔍 View document information"
    ):

        rows = []

        for item in (
            st.session_state.tender_pages
            + st.session_state.company_pages
        ):

            rows.append(
                {
                    "Document": item["document"],
                    "Type": item["document_type"],
                    "Page": item["page"],
                }
            )

        st.dataframe(
            rows,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# CHAT
# ============================================================

if st.session_state.documents_ready:

    st.divider()

    st.header(
        "💬 2. Ask BidReady AI"
    )

    st.caption(
        "Ask about suitability, requirements, "
        "gaps, experience, certifications, risks, "
        "or the recommended bid decision."
    )

    example_questions = [
        "Is this tender suitable for my company?",
        "What mandatory requirements do we fail?",
        "Which requirements do we satisfy?",
        "Do we have enough relevant experience?",
        "What certifications are required?",
        "What documents are missing?",
        "What are the biggest risks if we bid?",
        "Should we BID, BID WITH CONDITIONS, NO-BID, or NEEDS REVIEW?",
    ]

    selected_question = st.selectbox(
        "Example questions",
        ["Choose a question..."]
        + example_questions,
    )

    question = st.chat_input(
        "Ask BidReady AI..."
    )

    if (
        not question
        and selected_question != "Choose a question..."
    ):

        question = selected_question

    if question:

        st.chat_message(
            "user"
        ).write(
            question
        )

        with st.spinner(
            "Retrieving tender and company evidence..."
        ):

            try:

                embedding_model = (
                    st.session_state.embedding_model
                )

                tender_results = retrieve(
                    query=question,
                    index=st.session_state.tender_index,
                    chunks=st.session_state.tender_chunks,
                    embedding_model=embedding_model,
                    top_k=TOP_K_TENDER,
                )

                company_results = retrieve(
                    query=question,
                    index=st.session_state.company_index,
                    chunks=st.session_state.company_chunks,
                    embedding_model=embedding_model,
                    top_k=TOP_K_COMPANY,
                )

                answer = analyze_with_groq(
                    question=question,
                    tender_results=tender_results,
                    company_results=company_results,
                )

                st.session_state.messages.append(
                    {
                        "question": question,
                        "answer": answer,
                    }
                )

                st.chat_message(
                    "assistant"
                ).markdown(
                    answer
                )

                display_evidence(
                    "🔎 Retrieved Tender Evidence",
                    tender_results,
                )

                display_evidence(
                    "🏢 Retrieved Company Evidence",
                    company_results,
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
    "BidReady AI is an AI-assisted decision-support system. "
    "Always verify important requirements against the original "
    "tender documentation before making a final bid decision."
)
