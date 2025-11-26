# Architecture & Design

This document describes the architecture of the Gemini PDF Chatbot, including data flow, metadata formats, operational considerations, and recommended improvements.

## High-level overview
The system ingests user-uploaded PDFs, extracts per-page text and splits it into overlapping chunks, computes embeddings for each chunk using Google Generative AI, stores vectors in a FAISS index with associated provenance metadata, and answers user questions by retrieving top-K chunks and invoking Gemini via LangChain.

## Components
1. Ingest (Streamlit UI) — file uploader
2. Preprocess — PDF extraction (PyPDF2) and chunking (RecursiveCharacterTextSplitter)
3. Embeddings — Google Generative AI embeddings (or local alternatives)
4. Vector store — FAISS (local) with optional S3 sync for online availability
5. Retrieval + QA — similarity_search -> assemble prompt -> invoke Gemini LLM
6. Contact handling — contact form -> save to user_info.csv

## Metadata schema
Example chunk metadata:
{
  "id": "ingest-20251125-0001-chunk-45",
  "source": "invoice_2025-10.pdf",
  "page": 12,
  "chunk_index": 3,
  "char_range": [254, 1523],
  "created_at": "2025-11-25T12:34:56Z"
}

## Security & compliance
- Keep `GOOGLE_API_KEY` and AWS credentials out of git
- Consider local embedding for sensitive documents
- Use IAM roles for S3 and fine-grained secrets

## Future improvements
- Replace local FAISS + S3 sync with managed vector DB for multi-writer support
- Add incremental indexing
- Add PDF preview with page links for provenance
- Add proper intent detection for contact flows
