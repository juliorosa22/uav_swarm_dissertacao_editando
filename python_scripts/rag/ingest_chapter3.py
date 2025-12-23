from pathlib import Path
import re

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

CHAPTER_PATH = "03_rev_literatura.tex"
DB_PATH = "chroma_db"


def remove_latex_comments(text: str) -> str:
    """
    Remove LaTeX comments (%) but preserves escaped percent signs (\%).
    """
    cleaned_lines = []

    for line in text.splitlines():
        # Remove comments that start with % but not \%
        line = re.sub(r'(?<!\\)%.*', '', line)
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def ingest_chapter():
    raw_text = Path(CHAPTER_PATH).read_text(encoding="utf-8")

    # 🔹 Remove LaTeX comments
    text = remove_latex_comments(raw_text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150
    )

    chunks = splitter.split_text(text)

    embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectordb = Chroma(
        collection_name="rsl",
        embedding_function=embeddings,
        persist_directory=DB_PATH
    )

    vectordb.add_texts(
        texts=chunks,
        metadatas=[{"source": "chapter3"} for _ in chunks]
    )

    vectordb.persist()

    print(f"[OK] Capítulo 3 ingerido sem comentários ({len(chunks)} chunks)")


if __name__ == "__main__":
    ingest_chapter()
