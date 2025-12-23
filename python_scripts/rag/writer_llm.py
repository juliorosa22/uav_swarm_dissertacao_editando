import os
from dotenv import load_dotenv
import requests

from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings

# =========================
# CONFIG
# =========================
DB_PATH = "chroma_db"
COLLECTION_NAME = "rsl"
TOP_K = 6

load_dotenv()

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
API_URL = "https://api.perplexity.ai/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
    "Content-Type": "application/json"
}


# =========================
# RAG CONTEXT RETRIEVAL
# =========================
def retrieve_context(question: str, k: int = TOP_K) -> str:
    embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectordb = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=DB_PATH
    )

    docs = vectordb.similarity_search(question, k=k)

    context_blocks = []
    for d in docs:
        src = d.metadata.get("filename", d.metadata.get("source", "unknown"))
        context_blocks.append(
            f"[Source: {src}]\n{d.page_content}"
        )

    return "\n\n".join(context_blocks)


# =========================
# PERPLEXITY WRITER
# =========================
def ask_perplexity(question: str, context: str) -> str:
    prompt = f"""
You are assisting in revising the literature review of a master's dissertation.

The user question is written in English.
The output MUST be written in Brazilian Portuguese (academic tone).

IMPORTANT RULES:
- Use ONLY the provided context.
- Do NOT introduce new references.
- Do NOT fabricate citations.
- Do NOT assume information not present in the context.
- If needed, suggest improved wording or a candidate paragraph.

Context:
{context}

Task:
{question}

Write the answer in Brazilian Portuguese.
"""

    payload = {
        "model": "sonar-pro",
        "messages": [
            {"role": "system", "content": "You are a careful academic writing assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }

    response = requests.post(API_URL, headers=HEADERS, json=payload)
    response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    question = (
        """Based on the reviewed works in the provided context, classify each study according
to the level at which artificial intelligence is applied:
low-level control (LL), mid-level control (ML), or high-level decision-making (HL).

For each classification, briefly justify the choice based on the described
action space and system architecture.

Return the result as a structured list suitable for conversion into a LaTeX table."""
    )

    context = retrieve_context(question)
    answer = ask_perplexity(question, context)

    print("\n=== TEXTO GERADO (PT-BR) ===\n")
    print(answer)
