import os
import re
import csv
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import load_qa_chain
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# ----------------------
# Configuration / Setup
# ----------------------
load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("Google API key is missing. Please set GOOGLE_API_KEY in your .env file")

# Configure Google GenAI client
genai.configure(api_key=google_api_key)

# AWS / S3 configuration (for online FAISS index storage)
S3_BUCKET = os.getenv("S3_BUCKET")  # set this to enable S3-backed FAISS index
S3_PREFIX = os.getenv("S3_PREFIX", "faiss_index")
S3_REGION = os.getenv("S3_REGION", None)

# Constants
FAISS_INDEX_DIR = "faiss_index"
EMBEDDING_MODEL = "models/embedding-001"
EMBEDDING_CHUNK_SIZE = 1000
EMBEDDING_OVERLAP = 200
SIMILARITY_K = 4

# ----------------------
# Helpers / Caching
# ----------------------
@st.cache_resource
def get_embeddings():
    """Return a cached embeddings instance (cloud by default)."""
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)  # type: ignore

@st.cache_resource
def get_llm():
    """Return a cached Chat model instance."""
    return ChatGoogleGenerativeAI(model="gemini-pro", client=genai, temperature=0.2)

# ----------------------
# S3 helpers for online FAISS storage
# ----------------------
def s3_client():
    """Create a boto3 S3 client using env credentials or instance role."""
    kwargs = {}
    if S3_REGION:
        kwargs["region_name"] = S3_REGION
    return boto3.client("s3", **kwargs)

def upload_dir_to_s3(local_dir, bucket, prefix="faiss_index"):
    client = s3_client()
    uploaded = []
    for root, dirs, files in os.walk(local_dir):
        for fname in files:
            local_path = os.path.join(root, fname)
            # compute s3 key
            relative_path = os.path.relpath(local_path, local_dir)
            s3_key = os.path.join(prefix, relative_path).replace("\\", "/")
            try:
                client.upload_file(local_path, bucket, s3_key)
                uploaded.append(s3_key)
            except NoCredentialsError:
                raise
            except ClientError as e:
                raise
    return uploaded

def download_dir_from_s3(local_dir, bucket, prefix="faiss_index"):
    client = s3_client()
    paginator = client.get_paginator("list_objects_v2")
    found = False
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            key = obj["Key"]
            # compute relative path inside local_dir
            rel_path = os.path.relpath(key, prefix)
            if rel_path.startswith(".."):
                # safety: skip weird keys
                continue
            local_path = os.path.join(local_dir, rel_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            try:
                client.download_file(bucket, key, local_path)
                found = True
            except ClientError:
                raise
    return found

# ----------------------
# PDF ingestion & chunking
# ----------------------
def extract_text_and_chunks(pdf_files):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=EMBEDDING_CHUNK_SIZE, chunk_overlap=EMBEDDING_OVERLAP
    )

    all_chunks = []
    all_metadatas = []

    for pdf in pdf_files:
        try:
            reader = PdfReader(pdf)
        except Exception as e:
            st.warning(f"Could not read {getattr(pdf, 'name', str(pdf))}: {e}")
            continue

        num_pages = len(reader.pages)
        for page_idx in range(num_pages):
            page = reader.pages[page_idx]
            page_text = page.extract_text() or ""
            if not page_text.strip():
                continue

            chunks = splitter.split_text(page_text)
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                metadata = {
                    "source": getattr(pdf, "name", "uploaded_pdf"),
                    "page": page_idx + 1,
                    "chunk": chunk_idx + 1,
                }
                all_metadatas.append(metadata)

    return all_chunks, all_metadatas

# ----------------------
# Vector store management (local + optional S3 sync)
# ----------------------
def build_vector_store(texts, metadatas, upload_to_s3=False):
    embeddings = get_embeddings()
    vector_store = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)

    # ensure local dir exists
    if not os.path.exists(FAISS_INDEX_DIR):
        os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

    vector_store.save_local(FAISS_INDEX_DIR)

    # if S3 configured and user wants online storage, upload
    if upload_to_s3 and S3_BUCKET:
        try:
            uploaded = upload_dir_to_s3(FAISS_INDEX_DIR, S3_BUCKET, S3_PREFIX)
            st.info(f"Uploaded {len(uploaded)} index files to S3 bucket {S3_BUCKET}/{S3_PREFIX}")
        except NoCredentialsError:
            st.error("AWS credentials not found. Could not upload FAISS index to S3.")
        except Exception as e:
            st.error(f"Failed to upload FAISS index to S3: {e}")

    return vector_store

def load_vector_store(download_from_s3_if_missing=True):
    embeddings = get_embeddings()

    # If local index missing but S3 is configured, try download
    if not os.path.exists(FAISS_INDEX_DIR) or not os.listdir(FAISS_INDEX_DIR):
        if S3_BUCKET and download_from_s3_if_missing:
            try:
                os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
                found = download_dir_from_s3(FAISS_INDEX_DIR, S3_BUCKET, S3_PREFIX)
                if not found:
                    return None
            except NoCredentialsError:
                st.error("AWS credentials not found. Could not download FAISS index from S3.")
                return None
            except Exception as e:
                st.error(f"Failed to download FAISS index from S3: {e}")
                return None
        else:
            return None

    try:
        return FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Failed to load FAISS index: {e}")
        return None

# ----------------------
# Conversational chain
# ----------------------
def get_conversational_chain():
    prompt_template = (
        """
Answer the question as precisely and completely as possible using ONLY the provided context. If
an answer cannot be found in the context, reply exactly: "answer is not available in the context".
Do not hallucinate or invent facts. When returning an answer, include a short 'Sources' section listing
which PDF filename(s) and page(s) the information came from.

Context:
{context}

Question:
{question}

Answer:
        """
    )

    llm = get_llm()
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)
    return chain

# ----------------------
# Query handling
# ----------------------
def answer_question(question, k=SIMILARITY_K):
    vector_store = load_vector_store()
    if vector_store is None:
        return "Please upload and process PDF files first."

    try:
        docs = vector_store.similarity_search(question, k=k)
    except Exception as e:
        st.error(f"Error during similarity search: {e}")
        return "An error occurred while searching the index."

    chain = get_conversational_chain()
    try:
        result = chain.run(input_documents=docs, question=question)
    except TypeError:
        resp = chain.invoke({"input_documents": docs, "question": question})
        if isinstance(resp, dict):
            result = resp.get("output_text") or str(resp)
        else:
            result = str(resp)

    # Append sources
    try:
        sources = set()
        for d in docs:
            md = getattr(d, "metadata", {}) or {}
            src = md.get("source")
            page = md.get("page")
            if src:
                sources.add(f"{src}:page_{page}")
        if sources:
            result = f"{result}\n\nSources: {', '.join(sorted(sources))}"
    except Exception:
        pass

    return result

# ----------------------
# Utilities
# ----------------------
def save_user_info(name, phone, email):
    file_exists = os.path.isfile("user_info.csv")
    with open("user_info.csv", mode="a", newline="", encoding="utf-8") as f:
        fieldnames = ["Name", "Phone", "Email"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({"Name": name, "Phone": phone, "Email": email})

def validate_email(email):
    return bool(re.match(r"[^@]+@[^@]+\.[^@]+", email)) if email else True

def validate_phone(phone):
    return bool(re.match(r"^[\d\+\-\s]{6,20}$", phone)) if phone else True

# ----------------------
# Streamlit app
# ----------------------
def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "Upload some PDFs and ask me a question"}
    ]

def main():
    st.set_page_config(page_title="Gemini PDF Chatbot (S3 FAISS)", page_icon="🖐", layout="wide")

    # Sidebar: upload and process PDFs
    with st.sidebar:
        st.title("Menu")
        pdf_files = st.file_uploader(
            "Upload your PDF files (PDF only) and click 'Submit & Process'",
            type=["pdf"],
            accept_multiple_files=True,
        )

        upload_to_s3 = False
        if S3_BUCKET:
            upload_to_s3 = st.checkbox("Upload FAISS index to S3 (make it online)", value=True)

        if st.button("Submit & Process"):
            if not pdf_files:
                st.warning("Please upload at least one PDF file.")
            else:
                with st.spinner("Processing PDFs and building index..."):
                    texts, metadatas = extract_text_and_chunks(pdf_files)
                    if not texts:
                        st.warning("No extractable text found in uploaded PDFs.")
                    else:
                        build_vector_store(texts, metadatas, upload_to_s3=upload_to_s3)
                        st.success("Index built and saved.")

    # Main UI
    st.title("Chat with your PDF files")
    st.write("Ask questions about the uploaded PDF content. I will answer only from the provided PDFs.")
    st.sidebar.button("Clear Chat History", on_click=clear_chat_history)

    if "messages" not in st.session_state:
        clear_chat_history()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(f"**{message['role'].capitalize()}:** {message['content']}")

    # Chat input
    if prompt := st.chat_input(placeholder="Ask a question about your PDF files..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(f"**User:** {prompt}")

        # call me intent handling
        if "call me" in prompt.lower():
            st.session_state.collecting_info = True
            response = "Sure — please provide your contact details in the form that appears on the page."
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(f"**Assistant:** {response}")
            st.experimental_rerun()

        # ensure vector store exists
        elif load_vector_store() is None:
            response = "Please upload and process your PDF files first before asking a question."
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(f"**Assistant:** {response}")

        else:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        answer = answer_question(prompt)
                    except Exception as e:
                        answer = f"An error occurred while generating an answer: {e}"
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.markdown(f"**Assistant:** {answer}")

    # Contact form when "call me" intent triggered
    if st.session_state.get("collecting_info"):
        st.subheader("Please provide your contact details:")
        with st.form(key="contact_form"):
            name = st.text_input("Name")
            phone = st.text_input("Phone Number")
            email = st.text_input("Email")
            submit_button = st.form_submit_button(label="Submit")

            if submit_button:
                if not name or not phone:
                    st.warning("Please fill out at least Name and Phone Number.")
                elif not validate_phone(phone):
                    st.warning("Please enter a valid phone number.")
                elif not validate_email(email):
                    st.warning("Please enter a valid email address.")
                else:
                    save_user_info(name, phone, email)
                    response_text = f"Thank you, {name}. We will contact you at {phone} or {email}."
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                    st.session_state.collecting_info = False
                    st.success(response_text)
                    st.experimental_rerun()

if __name__ == "__main__":
    main()
