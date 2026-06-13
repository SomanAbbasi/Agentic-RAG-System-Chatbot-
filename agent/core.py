
import os
from dotenv import load_dotenv
import numexpr as ne

from langchain_groq import ChatGroq
from langchain.agents import create_agent
#create_agent replaces legacy AgentExecutor and create_react_agent
#from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import tool
#from langchain import hub

import wikipedia
# from langchain_community.tools import WikipediaQueryRun
# from langchain_community.utilities import WikipediaAPIWrapper

# from langchain_core.chat_history import InMemoryChatMessageHistory
# from langchain_core.runnables.history import RunnableWithMessageHistory
from langgraph.checkpoint.memory import MemorySaver
from prompts import system_prompt

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
        import wikipedia
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
    
    return [calculator_tool,wikipedia_search]
    

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
     

if __name__ == "__main__":
    agent = build_agent("llama-3.1-8b-instant", 0.0,checkpointer=MemorySaver())
    
    # LangGraph agents require a thread_id inside the execution config
    config = {"configurable": {"thread_id": "user_chat_session_45"}}
    
    # Run user prompt 1
    response = agent.invoke(
        {"messages": "Hi, I am Tony. What is 5 to the power of 3?"}, 
        config=config
    )
    print("Agent:", response["messages"][-1].content)
    
    response2 = agent.invoke(
        {"messages": "What is my name?"}, 
        config=config
    )
    print("Agent:", response2["messages"][-1].content)
     
     
    
