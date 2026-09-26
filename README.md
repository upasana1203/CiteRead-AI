# CiteRead AI

A Retrieval-Augmented Generation (RAG) app that lets you upload a PDF and chat with it. Every answer is grounded in the document and comes with inline citations pointing to the exact page and file it was pulled from — no unsourced claims.

## What it does

- Upload any PDF (research papers, reports, legal docs, manuals, resumes, etc.)
- Ask questions in a chat interface, or use the one-click suggested prompts (Summarize, Key Insights, Main Topics)
- Get answers grounded in the document, with inline citations like `(according to page 4 in report.pdf)`
- If the answer isn't in the document, the app says so instead of guessing
- Download the full conversation as a text file
- Inspect exactly which chunks of the document were retrieved for any answer

## How it works

1. **Text extraction** — the PDF is parsed page by page with `pypdf`. If a PDF has no extractable text (e.g. a scanned document with no OCR layer), the app tells you clearly instead of silently producing empty results.
2. **Chunking** — extracted text is split into overlapping chunks (1000 characters, 150-character overlap) while preserving which file and page each chunk came from.
3. **Embedding** — each chunk is embedded using the `all-MiniLM-L6-v2` Sentence-Transformer model.
4. **Vector search** — embeddings are indexed with FAISS for fast semantic similarity search.
5. **Retrieval** — on each question, the top-k most relevant chunks are retrieved.
6. **Generation** — retrieved chunks are passed to Gemini 2.5 Flash, which is instructed to answer only from the provided context and cite the source page/file for every factual claim.

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| PDF Parsing | pypdf |
| Embeddings | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| Vector Store | FAISS |
| LLM | Google Gemini 2.5 Flash |
| Language | Python |

## Setup

```bash
git clone https://github.com/<your-username>/CiteRead-AI.git
cd CiteRead-AI
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_api_key_here
```

Get a free Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

Run it locally:

```bash
streamlit run app.py
```

## Usage

1. Upload a PDF from the main page.
2. Once processed, ask a question in the chat box, or click one of the suggested prompts.
3. Read the answer and expand "Retrieved Context" to see exactly which chunks it was grounded in.
4. Use the sidebar to download the full conversation, or reset the session to start over with a new document.

## Project Structure

```
.
├── app.py              # Streamlit UI and app flow
├── utils.py             # PDF extraction, chunking, embedding, retrieval, generation
├── requirements.txt      # Dependencies
├── runtime.txt            # Python runtime version (for deployment)
├── .streamlit/
│   └── config.toml         # Upload size limit
└── extras/                  # Additional/experimental scripts
```

## Notes

- Only read-only, grounded question-answering is supported — the app never generates content that isn't backed by the uploaded document.
- Scanned or image-only PDFs without a text layer aren't currently supported (no OCR step yet); the app surfaces a clear error in that case instead of failing silently.
- A per-session cooldown between questions and a capped upload size (25MB) are in place to keep API usage and processing time reasonable on a free-tier deployment.

## Live Demo

🔗 *[Add your deployed Streamlit link here once live]*
