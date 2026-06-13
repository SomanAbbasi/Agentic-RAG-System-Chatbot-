import uuid
import streamlit as st
from langchain.messages import HumanMessage,SystemMessage,AIMessage

from agent.core import get_memory,build_agent

#Page Config

st.set_page_config(
    page_title="AI Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


#Session State  Bootstrap
#Runs once per browser session. Preserves history across reruns
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "checkpointer" not in st.session_state:
    st.session_state.checkpointer = get_memory()
    
# if "thinking" not in st.session_state:
#     st.session_state.thinking = False
    

# Sidebar

with st.sidebar:
    st.title("AI Agent")
    st.caption("Multi-agent · RAG · Web search")
    
    st.divider()
    
    #Model selector placeholder
    
    model = st.selectbox(
        "Model",
        ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"],
    )
    
    #st.slider(label, min_value, max_value, value, step)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    
    st.divider()
    
    if st.button("Clear Chat",use_container_width=True):
        st.session_state.messages = []
        st.session_state.checkpointer=get_memory() #reset memory
        st.session_state.thread_id=str(uuid.uuid4()) #reset thread_id for new conversation
        
        st.rerun()
        
    st.divider()
    st.caption(f"Thread: `{st.session_state.thread_id[:8]}...`")
    st.caption("Phase 3 — ReAct agent live")
        
    
#Main Chat Area

st.title("AI Agent")


# Render all existing messages from session state
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        

    

    
    
#Chat input handler 

if prompt := st.chat_input("Type your message here..."):
    
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    #Stream assistant response
    with st.chat_message("assistant"):
        
        
        with st.spinner("Thinking..."):
            agent=build_agent(
                model=model,
                temperature=temperature,
                checkpointer=st.session_state.checkpointer,
            )
            
            # thread_id in config tells MemorySaver which
            # conversation to load and save to
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            result = agent.invoke(
                {"messages": [HumanMessage(content=prompt)]},
                config=config,
            )
            
        response = result["messages"][-1].content
        st.markdown(response)
        
        tool_messages = [
            m for m in result["messages"]
            if hasattr(m, "type") and m.type == "tool"
        ]
        
        if tool_messages:
            with st.expander("Agent tool calls", expanded=False):
                for m in tool_messages:
                    st.markdown(f"**Tool:** `{m.name}`")
                    st.markdown(f"**Result:** {m.content}")
                    
    st.session_state.messages.append({"role": "assistant", "content": response})
        
        
        
        
        
        
        
    
    
    
    
