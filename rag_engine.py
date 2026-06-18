import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        print(f"PDF extraction error: {e}")
        return ""


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap
    return chunks


def get_gemini_embeddings(texts: list[str]) -> list[list[float]]:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GEMINI_API_KEY)
    embeddings = []
    for text in texts:
        result = client.models.embed_content(
            model="models/gemini-embedding-001",
            contents=types.Content(
                parts=[types.Part(text=text)]
            )
        )
        embeddings.append(result.embeddings[0].values)
    return embeddings


def build_vectorstore(pdf_path: str):
    try:
        import faiss
        import numpy as np

        text = extract_text_from_pdf(pdf_path)
        if not text:
            return None

        chunks = chunk_text(text)
        if not chunks:
            return None

        print(f"Created {len(chunks)} chunks")

        embeddings = get_gemini_embeddings(chunks)
        embeddings_np = np.array(embeddings, dtype="float32")

        dim = embeddings_np.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings_np)

        return {
            "index": index,
            "chunks": chunks,
            "embeddings": embeddings_np
        }

    except Exception as e:
        print(f"Vectorstore build error: {e}")
        return None


def retrieve_relevant_chunks(query: str, vectorstore: dict, top_k: int = 4) -> list[str]:
    from google import genai
    from google.genai import types
    import numpy as np

    client = genai.Client(api_key=GEMINI_API_KEY)
    result = client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=types.Content(
            parts=[types.Part(text=query)]
        )
    )
    query_embedding = np.array([result.embeddings[0].values], dtype="float32")

    distances, indices = vectorstore["index"].search(query_embedding, top_k)
    chunks = [vectorstore["chunks"][i]
              for i in indices[0] if i < len(vectorstore["chunks"])]
    return chunks


def query_rag(query: str, vectorstore: dict, chat_history: list = []) -> str:
    try:
        from groq import Groq

        relevant_chunks = retrieve_relevant_chunks(query, vectorstore)
        context = "\n\n---\n\n".join(relevant_chunks)

        history_str = ""
        for msg in chat_history[-4:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_str += f"{role}: {msg['content']}\n"

        system_prompt = """You are LexAI, an expert legal document assistant.
Your job is to answer questions about legal documents clearly and accurately.
- Base your answers ONLY on the provided document context
- Use simple, plain English — avoid unnecessary jargon
- If the answer isn't in the document, say so clearly
- For legal clauses, cite which part of the document supports your answer
- Be concise but thorough"""

        user_prompt = f"""Document Context:
{context}

{f'Previous conversation:{chr(10)}{history_str}' if history_str else ''}

Question: {query}

Answer based on the document:"""

        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=1024
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error generating answer: {str(e)}"
