import streamlit as st

from utils import (
    extract_text_from_pdf,
    chunk_text,
    create_vector_store,
    retrieve_relevant_chunks,
    generate_answer
)


# ======================
# PAGE CONFIG
# ======================

st.set_page_config(
    page_title="Document Intelligence System",
    page_icon="📚",
    layout="wide"
)


# ======================
# SIDEBAR
# ======================

with st.sidebar:

    st.title("📚 DocuMind AI")

    st.markdown(
        """
### AI-Powered Document Intelligence

Upload PDFs and interact with them using:

- RAG Pipeline
- FAISS Vector Search
- Semantic Retrieval
- Gemini LLM
- Context-Grounded Responses
"""
    )

    st.divider()

    st.markdown("### Supported Documents")

    st.markdown(
        """
- Research Papers
- Policy Documents
- Legal PDFs
- Technical Manuals
- Reports
- Resumes
"""
    )

    st.divider()

    if st.button("🔄 Reset Session"):

        st.session_state.messages = []

        st.session_state.document_processed = False

        st.rerun()


# ======================
# MAIN HEADER
# ======================

st.title("📚 Document Intelligence System")

st.markdown(
    """
Chat with your documents using Retrieval-Augmented Generation (RAG).
"""
)


# ======================
# SESSION STATE
# ======================

if "messages" not in st.session_state:

    st.session_state.messages = []

if "document_processed" not in st.session_state:

    st.session_state.document_processed = False


# ======================
# FILE UPLOAD
# ======================

uploaded_file = st.file_uploader(
    "Upload PDF Document",
    type="pdf"
)


# ======================
# DOCUMENT PROCESSING
# ======================

if (
    uploaded_file
    and not st.session_state.document_processed
):

    with st.spinner(
        "Processing document..."
    ):

        text = extract_text_from_pdf(
            uploaded_file
        )

        chunks = chunk_text(text)

        index = create_vector_store(
            chunks
        )

        st.session_state.chunks = chunks

        st.session_state.index = index

        st.session_state.document_processed = True

    st.success(
        "✅ Document processed successfully!"
    )

    st.info(
        f"""
Document split into
{len(chunks)} semantic chunks.
"""
    )


# ======================
# SUGGESTED QUESTIONS
# ======================

query = None

if st.session_state.document_processed:

    st.markdown("### Suggested Questions")

    col1, col2, col3 = st.columns(3)

    with col1:

        summary_btn = st.button(
            "📝 Summarize Document"
        )

    with col2:

        key_points_btn = st.button(
            "📌 Key Insights"
        )

    with col3:

        topics_btn = st.button(
            "🧠 Main Topics"
        )

    if summary_btn:

        query = "Summarize this document"

    elif key_points_btn:

        query = (
            "What are the key points "
            "in this document?"
        )

    elif topics_btn:

        query = (
            "What are the main topics "
            "discussed?"
        )


# ======================
# DISPLAY CHAT HISTORY
# ======================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ======================
# CHAT INPUT
# ======================

if st.session_state.document_processed:

    user_query = st.chat_input(
        "Ask questions about the document..."
    )

    if user_query:

        query = user_query


# ======================
# QUERY PROCESSING
# ======================

if (
    st.session_state.document_processed
    and query
):

    # USER MESSAGE
    st.chat_message(
        "user"
    ).markdown(query)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    # RETRIEVAL
    relevant_chunks = retrieve_relevant_chunks(
        query,
        st.session_state.index,
        st.session_state.chunks
    )

    # GENERATION
    with st.spinner(
        "Generating answer..."
    ):

        answer = generate_answer(
            query,
            relevant_chunks
        )

    # ASSISTANT RESPONSE
    with st.chat_message(
        "assistant"
    ):

        st.markdown(answer)

        with st.expander(
            "🔍 Retrieved Context"
        ):

            for i, chunk in enumerate(
                relevant_chunks
            ):

                st.markdown(
                    f"### Chunk {i+1}"
                )

                st.markdown(chunk)

                st.divider()

    # SAVE RESPONSE
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )