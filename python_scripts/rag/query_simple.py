from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings

DB_PATH = "chroma_db"

def ask_rag(question: str, k: int = 6):
    embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectordb = Chroma(
        collection_name="rsl",
        embedding_function=embeddings,
        persist_directory=DB_PATH
    )

    docs = vectordb.similarity_search(question, k=k)

    print("\n=== CONTEXTO RECUPERADO ===\n")
    for i, d in enumerate(docs, 1):
        meta = d.metadata
        src = meta.get("filename", meta.get("source"))
        print(f"[{i}] Fonte: {src}")
        print(d.page_content[:900])
        print("\n---\n")

if __name__ == "__main__":
    ask_rag(
        "Qual a quantidade de trabalhos existem na base?"
    )
