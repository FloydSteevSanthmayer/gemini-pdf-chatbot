# Gemini PDF Chatbot

A Streamlit-based chatbot that ingests PDF documents, builds an embeddings-based FAISS index, and answers questions using Google Generative AI (Gemini) via LangChain. Optionally uploads the FAISS index to S3 for shared/online access.

## Features
- Upload PDFs via Streamlit UI
- Per-page chunking with provenance metadata (filename, page)
- Embeddings via Google Generative AI (cloud)
- FAISS index stored locally and optionally synced to S3
- Chat interface powered by Gemini (via LangChain)
- Contact info capture (CSV)

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

## Deployment
- **Streamlit Cloud:** push the repo to GitHub and link to Streamlit Cloud, set environment variables there.
- **Docker:** use the provided `Dockerfile`.
- **Render / Heroku:** use `Procfile` (set env vars in service settings).

## Security & Compliance
- Do not commit `.env` or secret keys. Use provider secret stores.
- The app sends the uploaded text to Google to compute embeddings — ensure this complies with your data handling policy.
- Use an IAM role with limited S3 scope for the index upload.

## Troubleshooting
- If you see `ModuleNotFoundError: No module named 'langchain_core.pydantic_v1'` — uninstall the broken `langchain-google-genai==0.0.1` and reinstall a compatible package (see repo notes).

## License
MIT
