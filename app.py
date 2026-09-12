import os
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import fitz  # PyMuPDF
import streamlit as st
from dotenv import load_dotenv
from google import genai


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

APP_NAME = "BidReady AI"

PROMPT_FILE = Path(__file__).parent / "requirements_prompt.txt"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="BidReady AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM STYLING
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
            font-size: 1.2rem;
            color: #64748B;
            margin-bottom: 2rem;
        }

        .score-card {
            padding: 25px;
            border-radius: 16px;
            text-align: center;
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
        }

        .score-number {
            font-size: 3.5rem;
            font-weight: 800;
            margin: 0;
        }

        .recommendation {
            font-size: 1.4rem;
            font-weight: 700;
            margin-top: 8px;
        }

        .section-title {
            font-size: 1.5rem;
            font-weight: 700;
            color: #0F172A;
            margin-top: 1.5rem;
        }

        .risk-box {
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 10px;
        }

        .critical {
            background: #FEF2F2;
            border-left: 5px solid #DC2626;
        }

        .high {
            background: #FFF7ED;
            border-left: 5px solid #EA580C;
        }

        .medium {
            background: #FFFBEB;
            border-left: 5px solid #D97706;
        }

        .low {
            background: #F0FDF4;
            border-left: 5px solid #16A34A;
        }

        .stMetric {
            background-color: #F8FAFC;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_analysis_prompt() -> str:
    """
    Load the AI analysis instructions from requirements_prompt.txt.
    """

    if not PROMPT_FILE.exists():
        raise FileNotFoundError(
            "requirements_prompt.txt was not found."
        )

    return PROMPT_FILE.read_text(encoding="utf-8")


def extract_pdf_text(pdf_bytes: bytes) -> tuple[str, int]:
    """
    Extract text from a PDF while preserving page numbers.

    Returns:
        full_text: Combined text
        page_count: Number of pages
    """

    try:
        document = fitz.open(stream=pdf_bytes, filetype="pdf")

        pages = []

        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text")

            if text.strip():
                pages.append(
                    f"\n--- PAGE {page_number} ---\n{text.strip()}"
                )

        page_count = len(document)
        document.close()

        return "\n".join(pages), page_count

    except Exception as exc:
        raise RuntimeError(
            f"Unable to read the PDF: {exc}"
        ) from exc


def clean_json_response(response_text: str) -> str:
    """
    Remove Markdown code fences if Gemini returns JSON inside them.
    """

    text = response_text.strip()

    # Remove ```json ... ```
    text = re.sub(
        r"^```json\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Remove ``` ... ```
    text = re.sub(
        r"^```\s*",
        "",
        text,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    return text.strip()


def parse_ai_json(response_text: str) -> Dict[str, Any]:
    """
    Convert the AI response into a Python dictionary.
    """

    cleaned = clean_json_response(response_text)

    try:
        result = json.loads(cleaned)

        if not isinstance(result, dict):
            raise ValueError("AI response is not a JSON object.")

        return result

    except json.JSONDecodeError as exc:
        raise ValueError(
            "The AI returned an invalid JSON response."
        ) from exc


def get_gemini_client() -> genai.Client:
    """
    Create a Gemini API client.

    Supports:
        GEMINI_API_KEY
        GOOGLE_API_KEY
    """

    api_key = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )

    # Streamlit Cloud secrets support
    if not api_key:
        try:
            api_key = (
                st.secrets.get("GEMINI_API_KEY")
                or st.secrets.get("GOOGLE_API_KEY")
            )
        except Exception:
            api_key = None

    if not api_key:
        raise ValueError(
            "Gemini API key not found. "
            "Set GEMINI_API_KEY in your .env file "
            "or Streamlit secrets."
        )

    return genai.Client(api_key=api_key)


def analyze_tender(
    tender_text: str,
    company_profile: str,
    model_name: str,
) -> Dict[str, Any]:
    """
    Send the tender and company profile to Gemini
    and return structured analysis.
    """

    client = get_gemini_client()

    instructions = load_analysis_prompt()

    prompt = f"""
{instructions}

============================================================
TENDER DOCUMENT
============================================================

{tender_text}

============================================================
COMPANY PROFILE
============================================================

{company_profile}

============================================================
FINAL INSTRUCTION
============================================================

Analyze the tender against the company profile.

Return ONLY valid JSON.

Do not use Markdown.
Do not add explanations outside the JSON.
Do not invent company capabilities.
Do not invent tender requirements.
Use "NEEDS_REVIEW" when the available evidence is insufficient.

The JSON must follow the structure described in the
analysis instructions.
"""

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )

    if not response.text:
        raise ValueError(
            "Gemini returned an empty response."
        )

    return parse_ai_json(response.text)


def safe_score(value: Any) -> float:
    """
    Convert a value into a score between 0 and 100.
    """

    try:
        number = float(value)
        return max(0.0, min(100.0, number))
    except (TypeError, ValueError):
        return 0.0


def recommendation_color(recommendation: str) -> str:
    """
    Return a color for the recommendation.
    """

    normalized = recommendation.upper()

    if normalized == "BID":
        return "#16A34A"

    if normalized == "BID WITH CONDITIONS":
        return "#D97706"

    if normalized == "NO-BID":
        return "#DC2626"

    return "#64748B"


def display_score(score_data: Dict[str, Any]) -> None:
    """
    Display the overall readiness score.
    """

    score = safe_score(
        score_data.get("overall_score", 0)
    )

    recommendation = str(
        score_data.get(
            "recommendation",
            "NEEDS REVIEW",
        )
    )

    color = recommendation_color(recommendation)

    st.markdown(
        f"""
        <div class="score-card">
            <div style="color:#64748B;">
                BID READINESS
            </div>

            <div
                class="score-number"
                style="color:{color};"
            >
                {score:.0f}%
            </div>

            <div
                class="recommendation"
                style="color:{color};"
            >
                {recommendation}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_category_scores(
    score_data: Dict[str, Any]
) -> None:
    """
    Display category-level scores.
    """

    categories = score_data.get(
        "category_scores",
        [],
    )

    if not categories:
        return

    st.markdown(
        '<div class="section-title">Category Scores</div>',
        unsafe_allow_html=True,
    )

    columns = st.columns(
        min(len(categories), 3)
    )

    for index, category in enumerate(categories):

        column = columns[index % len(columns)]

        name = category.get(
            "category",
            "Unknown",
        )

        score = safe_score(
            category.get("score", 0)
        )

        column.metric(
            label=name,
            value=f"{score:.0f}%",
        )


def display_requirements(
    requirements: List[Dict[str, Any]],
    matches: List[Dict[str, Any]],
) -> None:
    """
    Display the compliance matrix.
    """

    st.markdown(
        '<div class="section-title">Compliance Matrix</div>',
        unsafe_allow_html=True,
    )

    if not requirements:
        st.info(
            "No requirements were identified."
        )
        return

    match_lookup = {
        str(item.get("requirement_id")): item
        for item in matches
    }

    rows = []

    for requirement in requirements:

        requirement_id = str(
            requirement.get(
                "requirement_id",
                "",
            )
        )

        match = match_lookup.get(
            requirement_id,
            {},
        )

        status = match.get(
            "status",
            "NEEDS_REVIEW",
        )

        rows.append(
            {
                "Requirement": requirement.get(
                    "title",
                    "Untitled",
                ),
                "Category": requirement.get(
                    "category",
                    "Other",
                ),
                "Mandatory": (
                    "Yes"
                    if requirement.get(
                        "mandatory",
                        False,
                    )
                    else "No"
                ),
                "Status": status,
                "Explanation": match.get(
                    "explanation",
                    "",
                ),
            }
        )

    st.dataframe(
        rows,
        use_container_width=True,
        hide_index=True,
    )


def display_risks(
    risks: List[Dict[str, Any]]
) -> None:
    """
    Display identified risks.
    """

    st.markdown(
        '<div class="section-title">Risks & Gaps</div>',
        unsafe_allow_html=True,
    )

    if not risks:
        st.success(
            "No significant risks were identified."
        )
        return

    for risk in risks:

        level = str(
            risk.get(
                "level",
                "MEDIUM",
            )
        ).upper()

        css_class = level.lower()

        title = risk.get(
            "title",
            "Untitled Risk",
        )

        description = risk.get(
            "description",
            "",
        )

        mitigation = risk.get(
            "mitigation",
            "",
        )

        st.markdown(
            f"""
            <div class="risk-box {css_class}">
                <strong>{level}: {title}</strong>
                <br>
                {description}

                {
                    f"<br><br><strong>Mitigation:</strong> {mitigation}"
                    if mitigation
                    else ""
                }
            </div>
            """,
            unsafe_allow_html=True,
        )


def display_action_plan(
    actions: List[Dict[str, Any]]
) -> None:
    """
    Display recommended actions.
    """

    st.markdown(
        '<div class="section-title">Action Plan</div>',
        unsafe_allow_html=True,
    )

    if not actions:
        st.info(
            "No additional actions were generated."
        )
        return

    for index, action in enumerate(
        actions,
        start=1,
    ):

        priority = action.get(
            "priority",
            "MEDIUM",
        )

        title = action.get(
            "title",
            "Action",
        )

        description = action.get(
            "description",
            "",
        )

        st.markdown(
            f"""
            **{index}. {title}**
            
            `{priority}` — {description}
            """
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
        Know Before You Bid — AI-powered tender intelligence
        and bid-readiness analysis.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    model_name = st.text_input(
        "Gemini Model",
        value="gemini-2.5-flash",
        help="Gemini model used for tender analysis.",
    )

    st.divider()

    st.markdown("### How it works")

    st.markdown(
        """
        1. Upload a tender PDF
        2. Enter company information
        3. BidReady AI extracts requirements
        4. Requirements are compared with the company
        5. Risks and gaps are identified
        6. A readiness score is calculated
        7. Bid/no-bid recommendation is generated
        """
    )

    st.divider()

    st.caption(
        "BidReady AI provides AI-assisted analysis "
        "and should not be treated as legal or official "
        "tender eligibility advice."
    )


# ============================================================
# INPUT SECTION
# ============================================================

st.header("1. Tender Document")

uploaded_file = st.file_uploader(
    "Upload tender PDF",
    type=["pdf"],
    help="Upload a searchable/text-based tender PDF.",
)


st.header("2. Company Profile")

company_profile = st.text_area(
    "Enter company information",
    height=250,
    placeholder=(
        "Example:\n\n"
        "Company Name: ABC Engineering Pvt Ltd\n"
        "Industry: Engineering & Construction\n"
        "Years of Experience: 7\n"
        "Certifications: ISO 9001, PEC Registration\n"
        "Past Projects: 3 government infrastructure projects\n"
        "Capabilities: Mechanical works, industrial installation, maintenance\n"
        "Financial Capacity: PKR 100 million annual turnover\n"
        "Geographic Coverage: Pakistan"
    ),
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

st.divider()

analyze_button = st.button(
    "🚀 Analyze Tender",
    type="primary",
    use_container_width=True,
)


# ============================================================
# ANALYSIS
# ============================================================

if analyze_button:

    if uploaded_file is None:

        st.error(
            "Please upload a tender PDF first."
        )

        st.stop()

    if not company_profile.strip():

        st.error(
            "Please enter the company profile."
        )

        st.stop()

    with st.spinner(
        "Extracting tender and analyzing requirements..."
    ):

        try:

            # --------------------------------------------
            # PDF EXTRACTION
            # --------------------------------------------

            pdf_bytes = uploaded_file.getvalue()

            tender_text, page_count = extract_pdf_text(
                pdf_bytes
            )

            if not tender_text.strip():

                st.error(
                    "No readable text was found in this PDF. "
                    "Please upload a searchable PDF."
                )

                st.stop()

            # --------------------------------------------
            # AI ANALYSIS
            # --------------------------------------------

            result = analyze_tender(
                tender_text=tender_text,
                company_profile=company_profile,
                model_name=model_name,
            )

            # Store result for the current session
            st.session_state["analysis"] = result
            st.session_state["page_count"] = page_count
            st.session_state["file_name"] = uploaded_file.name

        except Exception as exc:

            st.error(
                f"Analysis failed: {exc}"
            )

            st.info(
                "Check your Gemini API key, PDF, model name, "
                "and requirements_prompt.txt file."
            )

            st.stop()


# ============================================================
# DISPLAY RESULTS
# ============================================================

if "analysis" in st.session_state:

    result = st.session_state["analysis"]

    page_count = st.session_state.get(
        "page_count",
        0,
    )

    file_name = st.session_state.get(
        "file_name",
        "Tender",
    )

    st.divider()

    st.header("📊 Bid Readiness Analysis")

    st.caption(
        f"Document: {file_name} • "
        f"{page_count} pages"
    )

    # --------------------------------------------
    # EXECUTIVE SUMMARY
    # --------------------------------------------

    summary = result.get(
        "executive_summary"
    )

    if summary:

        st.markdown(
            '<div class="section-title">Executive Summary</div>',
            unsafe_allow_html=True,
        )

        st.info(summary)

    # --------------------------------------------
    # SCORE
    # --------------------------------------------

    score_data = result.get(
        "readiness_score",
        {},
    )

    if isinstance(score_data, dict):

        display_score(score_data)

        st.write("")

        display_category_scores(
            score_data
        )

        explanation = score_data.get(
            "explanation"
        )

        if explanation:

            st.markdown(
                "**Scoring Explanation:**"
            )

            st.write(explanation)

    # --------------------------------------------
    # KEY COUNTS
    # --------------------------------------------

    requirements = result.get(
        "requirements",
        [],
    )

    matches = result.get(
        "matches",
        [],
    )

    risks = result.get(
        "risks",
        [],
    )

    actions = result.get(
        "action_plan",
        [],
    )

    mandatory_gaps = score_data.get(
        "mandatory_gaps",
        0,
    )

    critical_risks = score_data.get(
        "critical_risks",
        0,
    )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Requirements",
        len(requirements),
    )

    col2.metric(
        "Matches",
        sum(
            1
            for item in matches
            if item.get("status") == "MATCH"
        ),
    )

    col3.metric(
        "Mandatory Gaps",
        mandatory_gaps,
    )

    col4.metric(
        "Critical Risks",
        critical_risks,
    )

    # --------------------------------------------
    # COMPLIANCE MATRIX
    # --------------------------------------------

    display_requirements(
        requirements,
        matches,
    )

    # --------------------------------------------
    # RISKS
    # --------------------------------------------

    display_risks(risks)

    # --------------------------------------------
    # ACTION PLAN
    # --------------------------------------------

    display_action_plan(actions)

    # --------------------------------------------
    # RAW JSON
    # --------------------------------------------

    with st.expander(
        "🔍 View Full AI Analysis (JSON)"
    ):

        st.json(result)

    # --------------------------------------------
    # DOWNLOAD
    # --------------------------------------------

    json_download = json.dumps(
        result,
        indent=2,
        ensure_ascii=False,
    )

    st.download_button(
        label="⬇️ Download Analysis",
        data=json_download,
        file_name="bidready_analysis.json",
        mime="application/json",
        use_container_width=True,
    )
