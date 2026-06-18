# ⚖️ LexAI — Legal Document Q&A System

A RAG-based legal document assistant that lets you upload any legal PDF and ask questions about it in plain English.

## Tech Stack
- **FAISS** — Vector similarity search
- **Google Gemini Embeddings** (`text-embedding-004`) — Document & query embeddings
- **Groq LLaMA 3** (`llama3-70b-8192`) — Answer generation
- **Streamlit** — Web UI
- **pypdf** — PDF text extraction

## Setup

### 1. Clone & install dependencies
```bash
git clone <your-repo-url>
cd lexai
pip install -r requirements.txt
```

### 2. Set up API keys
```bash
cp .env.example .env
# Edit .env and add your keys
```

Get your keys:
- **Gemini**: https://aistudio.google.com/app/apikey (free)
- **Groq**: https://console.groq.com/keys (free)

### 3. Run locally
```bash
streamlit run app.py
```

## Deploying to Render

1. Push to GitHub
2. Create new **Web Service** on Render
3. Build command: `pip install -r requirements.txt`
4. Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
5. Add environment variables: `GEMINI_API_KEY` and `GROQ_API_KEY`

## Project Structure
```
lexai/
├── app.py              # Streamlit UI
├── rag_engine.py       # RAG pipeline (embed, retrieve, generate)
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```
## Demo
https://lexai-rag-based-legal-document-qanda.onrender.com
