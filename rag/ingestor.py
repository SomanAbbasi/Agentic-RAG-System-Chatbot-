import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CHROMA_PATH = "chroma_db"
COLLECTION  = "agent_knowledge"


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )


def get_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )


def ingest(file_path: str) -> int:

    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".pdf":
        docs = PyPDFLoader(file_path).load()
    elif ext == ".txt":
        docs = TextLoader(file_path, encoding="utf-8").load()
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    chunks = get_splitter().split_documents(docs)

    Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_PATH,
    ).add_documents(chunks)

    return len(chunks)


def ingest_bytes(file_bytes: bytes, filename: str) -> int:
    ext = os.path.splitext(filename)[-1].lower()
    if ext not in (".pdf", ".txt"):
        raise ValueError(f"Unsupported file type: {ext}. Use .pdf or .txt")

    # Write bytes to a temp file, ingest, then clean up
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=ext
    ) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        return ingest(tmp_path)
    finally:
        os.unlink(tmp_path)


def get_ingested_files() -> list[str]:
    try:
        store = Chroma(
            collection_name=COLLECTION,
            embedding_function=get_embeddings(),
            persist_directory=CHROMA_PATH,
        )
        results = store.get(include=["metadatas"])
        sources = set()
        for meta in results["metadatas"]:
            src = meta.get("source", "")
            if src:
                sources.add(os.path.basename(src))
        return sorted(sources)
    except Exception:
        return []


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m rag.ingestor <file_path>")
        sys.exit(1)
    count = ingest(sys.argv[1])
    print(f"Stored {count} chunks in ChromaDB at '{CHROMA_PATH}'")