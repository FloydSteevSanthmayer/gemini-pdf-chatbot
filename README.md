# Gemini PDF Chatbot

A Streamlit-based chatbot that ingests PDF documents, builds an embeddings-based FAISS index, and answers questions using Google Generative AI (Gemini) via LangChain. Optionally uploads the FAISS index to S3 for shared/online access.

## Features
- Upload PDFs via Streamlit UI
- Per-page chunking with provenance metadata (filename, page)
- Embeddings via Google Generative AI (cloud)
- FAISS index stored locally and optionally synced to S3
- Chat interface powered by Gemini (via LangChain)
- Contact info capture (CSV)

## Flowchart / Architecture
![Flowchart: Gemini PDF Chatbot](assets/flowchart-gemini-pdf-chatbot.jpg)

## Short summary
A user uploads PDFs → the app extracts text and splits it into chunks → embeddings are created and a FAISS index is saved → the user asks a question via chat input. If the user asks “call me”, the app shows a contact form and saves details to `user_info.csv`. Otherwise the app loads the FAISS index, performs a similarity search, assembles a QA chain, invokes the Gemini LLM, and returns the answer in chat.

## Quickstart (local)
1. Clone repo
```bash
git clone https://github.com/<your-org>/<repo>.git
cd <repo>
```

2. Create `.env` from `.env.example` and fill values.

3. Create and activate virtualenv
```bash
python -m venv .venv
# PowerShell
.venv\\Scripts\\Activate.ps1
# macOS/Linux
source .venv/bin/activate
```

4. Install dependencies
```bash
pip install -r requirements.txt
# If needed, install langchain-google-genai from GitHub (see notes in requirements.txt)
```

5. Run the app
```bash
streamlit run gemini_pdf_chatbot.py
```

## License
MIT — see [LICENSE](LICENSE)

## Security & Compliance
- Do not commit `.env` or secret keys. Use provider secret stores.
- The app sends the uploaded text to Google to compute embeddings — ensure this complies with your data handling policy.
- Use an IAM role with limited S3 scope for the index upload.
