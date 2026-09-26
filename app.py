import time
import streamlit as st

from utils import (
    extract_text_from_pdf,
    chunk_text,
    create_vector_store,
    retrieve_relevant_chunks,
    generate_answer
)

# Minimum seconds a user must wait between two questions, so one session
# can't hammer the Gemini API quota.
MIN_SECONDS_BETWEEN_QUERIES = 5


def is_rate_limited() -> bool:
    last_call = st.session_state.get("last_query_time", 0)
    elapsed = time.time() - last_call
    if elapsed < MIN_SECONDS_BETWEEN_QUERIES:
        wait = round(MIN_SECONDS_BETWEEN_QUERIES - elapsed, 1)
        st.warning(f"Please wait {wait}s before asking another question.")
        return True
    st.session_state["last_query_time"] = time.time()
    return False


# ======================
# PAGE CONFIG
# ======================

st.set_page_config(
    page_title="CiteRead AI",
    page_icon="📖",
    layout="wide"
)


# ======================
# SIDEBAR
# ======================

with st.sidebar:

    st.title("📖 CiteRead AI")

    st.markdown(
        """
### AI-Powered Document Intelligence

Upload PDFs and interact with them using:

- RAG Pipeline
- FAISS Vector Search
- Semantic Retrieval
- Gemini LLM
- Context-Grounded Responses with Citations
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

st.title("📖 CiteRead AI")

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

        try:
            text = extract_text_from_pdf(
                uploaded_file
            )

            chunks = chunk_text(
                text,
                source_name=uploaded_file.name
            )

            index = create_vector_store(
                chunks
            )

            st.session_state.chunks = chunks

            st.session_state.index = index

            st.session_state.document_processed = True

        except ValueError as e:
            st.error(f"⚠️ {e}")
            st.session_state.document_processed = False

    if st.session_state.document_processed:

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

    if is_rate_limited():
        st.stop()

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
                    f"### Chunk {i+1} — {chunk['source']}, page {chunk['page']}"
                )

                st.markdown(chunk["text"])

                st.divider()

    # SAVE RESPONSE
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )


# ======================
# EXPORT CONVERSATION
# ======================

if st.session_state.document_processed and st.session_state.messages:

    with st.sidebar:

        st.divider()

        transcript = "\n\n".join(
            f"{m['role'].upper()}: {m['content']}"
            for m in st.session_state.messages
        )

        st.download_button(
            "⬇️ Download conversation",
            data=transcript.encode("utf-8"),
            file_name="citeread_conversation.txt",
            mime="text/plain",
        )