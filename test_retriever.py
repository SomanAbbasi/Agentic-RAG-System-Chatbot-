from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

db = Chroma(
    collection_name="agent_knowledge",
    persist_directory="chroma_db",
    embedding_function=HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    ),
)

query = "gender of hen"

results = db.similarity_search(query, k=3)

for i, doc in enumerate(results, 1):
    print(f"\nResult {i}")
    print(doc.page_content)