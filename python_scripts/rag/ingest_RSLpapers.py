from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import re

PAPERS_DIR = "artigos_RSL"
TEX_FILE = "03_rev_literatura.tex"
DB_PATH = "chroma_db"

def remove_tex_comments(text):
    """Remove LaTeX comments (lines starting with %)"""
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        # Remove inline comments but keep escaped percent signs
        if not line.strip().startswith('%'):
            # Remove comments after code but keep \%
            line = re.sub(r'(?<!\\)%.*$', '', line)
            if line.strip():  # Only add non-empty lines
                cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)

def ingest_papers():
    embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectordb = Chroma(
        collection_name="rsl",
        embedding_function=embeddings,
        persist_directory=DB_PATH
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    total_chunks = 0
    processed_pdfs = 0
    failed_pdfs = []

    # Process PDF files
    pdf_files = list(Path(PAPERS_DIR).glob("*.pdf"))
    print(f"[INFO] Found {len(pdf_files)} PDF files in {PAPERS_DIR}")
    
    for pdf_path in pdf_files:
        try:
            print(f"[INFO] Processing {pdf_path.name}")

            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()

            chunks = splitter.split_documents(pages)

            # Add metadata to each chunk
            for chunk in chunks:
                chunk.metadata["source"] = "paper"
                chunk.metadata["filename"] = pdf_path.name
                chunk.metadata["type"] = "pdf"

            vectordb.add_documents(chunks)
            total_chunks += len(chunks)
            processed_pdfs += 1
            print(f"[OK] Added {len(chunks)} chunks from {pdf_path.name}")
            
        except Exception as e:
            print(f"[ERROR] Failed to process {pdf_path.name}: {str(e)}")
            failed_pdfs.append(pdf_path.name)

    # Process TEX file
    tex_path = Path(TEX_FILE)
    if tex_path.exists():
        print(f"[INFO] Processing {tex_path.name}")
        
        with open(tex_path, 'r', encoding='utf-8') as f:
            tex_content = f.read()
        
        # Remove LaTeX comments
        cleaned_content = remove_tex_comments(tex_content)
        
        # Create a document-like structure
        tex_doc = Document(
            page_content=cleaned_content,
            metadata={
                "source": "literatura_review",
                "filename": tex_path.name,
                "type": "tex"
            }
        )
        
        # Split the tex content
        tex_chunks = splitter.split_documents([tex_doc])
        
        vectordb.add_documents(tex_chunks)
        total_chunks += len(tex_chunks)
        
        print(f"[INFO] Added {len(tex_chunks)} chunks from {tex_path.name}")
    else:
        print(f"[WARNING] TEX file not found: {tex_path}")

    print(f"\n[SUMMARY]")
    print(f"  Total PDFs processed: {processed_pdfs}/{len(pdf_files)}")
    print(f"  Total chunks ingested: {total_chunks}")
    if failed_pdfs:
        print(f"  Failed PDFs: {', '.join(failed_pdfs)}")

if __name__ == "__main__":
    ingest_papers()
