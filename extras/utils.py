from pypdf import PdfReader

from sentence_transformers import (
    SentenceTransformer
)

import faiss
import numpy as np

import google.generativeai as genai

import os

from dotenv import load_dotenv

import streamlit as st


# ======================
# LOAD ENV VARIABLES
# ======================

load_dotenv()


# ======================
# CONFIGURE GEMINI
# ======================

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

gemini_model = genai.GenerativeModel(
    "gemini-2.5-flash"
)


# ======================
# LOAD EMBEDDING MODEL
# ======================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


def get_embedding_model():

    return load_embedding_model()


# ======================
# PDF TEXT EXTRACTION
# ======================

def extract_text_from_pdf(pdf_file):

    reader = PdfReader(pdf_file)

    text = ""

    for page in reader.pages:

        extracted_text = page.extract_text()

        if extracted_text:

            text += extracted_text + "\n"

    return text


# ======================
# SMART CHUNKING
# ======================

def chunk_text(
    text,
    chunk_size=1000,
    overlap=150
):

    paragraphs = text.split("\n")

    cleaned_paragraphs = []

    for para in paragraphs:

        para = para.strip()

        if len(para) > 40:

            cleaned_paragraphs.append(para)

    chunks = []

    current_chunk = ""

    for para in cleaned_paragraphs:

        if (
            len(current_chunk) + len(para)
            < chunk_size
        ):

            current_chunk += (
                para + "\n"
            )

        else:

            chunks.append(
                current_chunk.strip()
            )

            current_chunk = para + "\n"

    if current_chunk:

        chunks.append(
            current_chunk.strip()
        )

    return chunks

# ======================
# VECTOR STORE CREATION
# ======================

def create_vector_store(chunks):

    embeddings = get_embedding_model().encode(
        chunks
    )

    embeddings = np.array(
        embeddings
    ).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(embeddings)

    return index


# ======================
# RETRIEVAL
# ======================

def retrieve_relevant_chunks(
    query,
    index,
    chunks,
    top_k=4
):

    query_embedding = get_embedding_model().encode(
        [query]
    )

    query_embedding = np.array(
        query_embedding
    ).astype("float32")

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    retrieved_chunks = []

    for idx in indices[0]:

        if idx < len(chunks):

            retrieved_chunks.append(
                chunks[idx]
            )

    return retrieved_chunks


# ======================
# ANSWER GENERATION
# ======================

def generate_answer(
    query,
    retrieved_chunks
):

    context = "\n\n".join(
        retrieved_chunks
    )

    prompt = f"""
You are an intelligent document analysis assistant.

Your task is to answer questions ONLY using
the provided document context.

Answer in 100-150 words and use more only if asked to answer in detail.

Rules:
- Do not hallucinate.
- Do not make up information.
- If answer is not present in context,
  say:
  "Information not found in the document."

Provide concise and accurate answers.

Document Context:
{context}

Question:
{query}

Answer:
"""

    try:

        response = gemini_model.generate_content(
            prompt
        )

        return response.text

    except Exception as e:

        return f"Error: {str(e)}"