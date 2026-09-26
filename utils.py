from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.generativeai as genai
import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-3.8-flash")

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding_model():
    return load_embedding_model()

def extract_text_from_pdf(pdf_file):
    """
    Returns a list of {"page": n, "text": ...} dicts for pages with extractable
    text. Raises ValueError if the PDF is unreadable or has no extractable text
    at all (e.g. a scanned/image-only PDF with no OCR layer), so the caller can
    show a clear message instead of silently proceeding with zero chunks.
    """
    try:
        reader = PdfReader(pdf_file)
    except Exception as e:
        raise ValueError(f"Could not read this PDF: {e}")

    if len(reader.pages) == 0:
        raise ValueError("This PDF has no pages.")

    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        extracted_text = page.extract_text()
        if extracted_text and extracted_text.strip():
            pages.append({"page": page_number, "text": extracted_text})

    if not pages:
        raise ValueError(
            "No extractable text found in this PDF. It may be a scanned "
            "or image-only document without a text layer (OCR would be "
            "needed to process it)."
        )
    return pages

def chunk_text(pages, chunk_size=1000, overlap=150, source_name="document.pdf"):
    """Create character-overlapping chunks while preserving source/page metadata."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    chunks = []
    for page_data in pages:
        page_number = page_data["page"]
        text = " ".join(page_data["text"].split())
        if not text:
            continue
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append({"text": chunk, "source": source_name, "page": page_number})
            if end >= len(text):
                break
            start = end - overlap
    return chunks

def create_vector_store(chunks):
    if not chunks:
        raise ValueError("No text chunks to index — the document may be empty.")
    texts = [chunk["text"] for chunk in chunks]
    embeddings = get_embedding_model().encode(texts)
    embeddings = np.array(embeddings).astype("float32")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def retrieve_relevant_chunks(query, index, chunks, top_k=4):
    if not chunks:
        return []
    query_embedding = get_embedding_model().encode([query])
    query_embedding = np.array(query_embedding).astype("float32")
    k = min(top_k, len(chunks))
    distances, indices = index.search(query_embedding, k)
    retrieved_chunks = []
    for rank, idx in enumerate(indices[0], start=1):
        if idx < len(chunks):
            chunk = chunks[idx].copy()
            chunk["rank"] = rank
            chunk["distance"] = float(distances[0][rank - 1])
            retrieved_chunks.append(chunk)
    return retrieved_chunks

def generate_answer(query, retrieved_chunks):
    if not retrieved_chunks:
        return "Information not found in the document."
    context = "\n\n".join(
        f"SOURCE [{i}]\nFile: {c['source']}\nPage: {c['page']}\nContent:\n{c['text']}"
        for i, c in enumerate(retrieved_chunks, start=1)
    )
    prompt = f"""
You are an intelligent document analysis assistant.

Answer questions ONLY using the provided document context.

Rules:
- Do not hallucinate or make up information.
- If the answer is not present in the context, say: "Information not found in the document."
- Every factual claim based on the document MUST have an inline citation.
- Use exactly this citation style: (according to page X in filename.pdf)
- Use the exact file name and page number provided in the context.
- Do not invent citations.
- Put citations immediately after the relevant sentence/claim.
- Answer in 100-150 words unless the user asks for detail.

Document Context:
{context}

Question:
{query}

Answer:
"""
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"
