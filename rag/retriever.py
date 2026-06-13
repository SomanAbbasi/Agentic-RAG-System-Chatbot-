
#This is the @tool the agent calls when it needs to search your documents
import os
from langchain.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CHROMA_PATH = "chroma_db"
COLLECTION  = "agent_knowledge"


def get_vector_store()->Chroma:
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH,
    )


@tool 
def rag_search(query: str) -> str:
    """
    ALWAYS use this tool FIRST before any other tool.
    Search the private knowledge base for relevant information.
    This contains uploaded documents, notes, and domain-specific content.
    Use this for ANY question before trying web search or Wikipedia.
    Input should be a clear question or topic to search for.
    
    """
    
    try:
        vector_store=get_vector_store()  
        # similarity_search returns the top-k most relevant chunks
        results = vector_store.similarity_search(query, k=3)
        
        if not results:
            return "No relevant information found in knowledge base."
        
        #Format results clearly so the agent can reason over them
        output=[]
        
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get("source", "unknown source")
            page= doc.metadata.get("page", "unknown page")
            label= f"[{i}] {os.path.basename(source)}"
            
            if page:
                label += f" (page {page})"
            output.append(f"{label}:\n{doc.page_content}")
        
        return "\n\n".join(output)
    except Exception as e:
        return f"RAG search error: {e}"
            
    
