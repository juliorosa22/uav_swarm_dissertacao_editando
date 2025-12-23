import os
import json
from datetime import datetime
from dotenv import load_dotenv

from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from openai import OpenAI

# =========================
# CONFIG
# =========================
DB_PATH = "chroma_db"
COLLECTION_NAME = "rsl"
TOP_K = 10
OUTPUT_JSON = "llm_interactions.json"

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================
# RAG: CONTEXT RETRIEVAL
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
            f"[Fonte: {src}]\n{d.page_content}"
        )

    return "\n\n".join(context_blocks)

# =========================
# OPENAI: WRITER
# =========================
def ask_openai(question: str, context: str) -> str:
    prompt = f"""
You are assisting in revising the literature review of a master's dissertation
in robotics and artificial intelligence.

The question is written in English.
The answer MUST be written in Brazilian Portuguese, using an academic tone.

STRICT RULES:
- Use ONLY the provided context.
- Do NOT introduce new references.
- Do NOT fabricate citations.
- Do NOT assume information not explicitly present.
- If appropriate, generate a concise paragraph suitable for direct inclusion
  in a dissertation chapter.

Context:
{context}

Task:
{question}

Write the answer in Brazilian Portuguese.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a careful academic writing assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    return response.choices[0].message.content

# =========================
# JSON PERSISTENCE
# =========================
def save_interaction(question: str, answer: str, context: str):
    """Save the interaction to a JSON file"""
    
    # Load existing data
    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = {"interactions": []}
    
    # Create new interaction entry
    interaction = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer,
        "context": context,
        "model": "gpt-4o-mini",
        "top_k": TOP_K
    }
    
    # Append to list
    data["interactions"].append(interaction)
    
    # Save back to file
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n[INFO] Interaction saved to {OUTPUT_JSON}")
    print(f"[INFO] Total interactions: {len(data['interactions'])}")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    question = """
    Analise a estrutura e a organização do Capítulo 3 (Trabalhos Relacionados)
com base exclusivamente no contexto fornecido.

Avalie:
- o encadeamento lógico entre seções e subseções;
- se a estrutura atual sustenta de forma clara as questões de pesquisa definidas;
- possíveis redundâncias ou sobreposições conceituais entre seções;
- elementos estruturais ausentes ou pouco explorados em uma revisão sistemática;
- oportunidades de melhorar clareza, coerência e legibilidade para a banca.

Não reescreva o capítulo.
Não introduza novos trabalhos ou referências.
Forneça apenas recomendações estruturais e sugestões de reorganização.

Apresente a resposta como uma lista estruturada de recomendações,
adequada para uma dissertação de mestrado.
"""

    context = retrieve_context(question)
    answer = ask_openai(question, context)

    print("\n=== TEXTO GERADO (PT-BR) ===\n")
    print(answer)
    
    # Save interaction
    save_interaction(question, answer, context)
