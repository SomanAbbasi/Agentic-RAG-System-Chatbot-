
import os
from langchain_community.document_loaders import PyPDFLoader,TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


CHROMA_PATH = "chroma_db"        # folder where vectors are stored on disk
COLLECTION  = "agent_knowledge"  # name of the collection inside ChromaDB


# Embeddings Model

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )


#Text splitter

def get_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    
# Loaders
def load_pdf(path: str):
    return PyPDFLoader(path).load()

def load_text(path: str):
    return TextLoader(path, encoding="utf-8").load()

def ingest(file_path: str):

    print(f"Loading: {file_path}")

    ext = os.path.splitext(file_path)[-1].lower()

    if ext == ".pdf":
        documents = load_pdf(file_path)

    elif ext == ".txt":
        documents = load_text(file_path)

    else:
        raise ValueError(
            f"Unsupported file type: {ext}"
        )

    splitter = get_splitter()

    chunks = splitter.split_documents(documents)

    print(f"Split into {len(chunks)} chunks")

    embeddings = get_embeddings()

    vector_store = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH,
    )

    vector_store.add_documents(chunks)

    print(
        f"Stored {len(chunks)} chunks in '{CHROMA_PATH}'"
    )
    
   
    
    
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m rag.ingestor <file_path>")
        sys.exit(1)
    ingest(sys.argv[1])
    

