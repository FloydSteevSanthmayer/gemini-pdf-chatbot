"""CLI helper: build a FAISS index from PDF files without running the Streamlit UI.
Usage: python scripts/build_index.py --pdfs path1.pdf path2.pdf --upload-s3
"""
import argparse
import os
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from PyPDF2 import PdfReader
EMBEDDING_MODEL = "models/embedding-001"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
FAISS_DIR = "faiss_index"
def extract_texts(pdf_paths):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    texts = []
    metadatas = []
    for p in pdf_paths:
        reader = PdfReader(p)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if not text.strip():
                continue
            chunks = splitter.split_text(text)
            for j, c in enumerate(chunks):
                texts.append(c)
                metadatas.append({"source": Path(p).name, "page": i+1, "chunk": j+1})
    return texts, metadatas
def build_index(pdf_paths, upload_to_s3=False):
    texts, metadatas = extract_texts(pdf_paths)
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vector_store = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
    os.makedirs(FAISS_DIR, exist_ok=True)
    vector_store.save_local(FAISS_DIR)
    print(f"Saved index to {FAISS_DIR}")
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdfs', nargs='+', required=True)
    parser.add_argument('--upload-s3', action='store_true')
    args = parser.parse_args()
    build_index(args.pdfs, upload_to_s3=args.upload_s3)
