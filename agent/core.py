
import os
import wikipedia
import numexpr as ne
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from .prompts import system_prompt

#from langchain import hub

load_dotenv()

#LLM

def get_llm(model:str,temperature:float):
    
    return ChatGroq(
        model=model,
        temperature=temperature,
        api_key=os.getenv("GROQ_API_KEY"),
        streaming=True
    )


# Tools
@tool 
def calculator_tool(expression: str) -> str:
    """
    Safe math evaluator using numexpr.

    Args:
        expression: Math expression like "2 ** 10" or "(4 + 5) * 3"

    Returns:
        Result as string
    """
    try:
        result = ne.evaluate(expression)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"
  
@tool
def wikipedia_search(query: str) -> str:
    """
    Search Wikipedia for factual information about a topic.
    Use this for definitions, historical facts, concepts, and
    background knowledge. Input should be a clear topic or question.
    """
    try:
        results = wikipedia.search(query, results=2)
        if not results:
            return "No Wikipedia results found."
        summary = wikipedia.summary(results[0], sentences=4)
        return f"Wikipedia — {results[0]}:\n{summary}"
    except Exception as e:
        return f"Wikipedia error: {e}"
    
def get_tools()->list:
    """
    Returns the list of tools available to the agent.
    """
    
    web_search = TavilySearch(
        max_results=3,
        include_answer=True,
        topic="general",
    )
    # Override description so agent knows when to use it
    web_search.description = (
        "Search the web for current, recent, or live information. "
        "Use for news, latest releases, prices, recent events, "
        "or anything that may have changed recently. "
        "Input should be a clear search query string."
    )
    
    return [calculator_tool,wikipedia_search,web_search]
    

# Memory

    

def get_memory():
    """
    Returns a ConversationBufferMemory instance for the agent.
    """
    return MemorySaver()

#Agent factory

def build_agent(model:str,temperature:float,checkpointer: MemorySaver):
     """
        Assembles the full ReAct agent.

        create_react_agent:  binds LLM + tools + prompt template together
        AgentExecutor:       the runtime that drives the Thought/Action/Observe loop
    """
    
     llm=get_llm(model,temperature)
     tools=get_tools()
     
     prompt = system_prompt()
     
     return create_agent(
         model=llm,
         tools=tools,
         system_prompt=prompt,
         checkpointer=checkpointer
     )
     


     
     
    
