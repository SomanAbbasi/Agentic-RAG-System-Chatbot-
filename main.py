from urllib import response

import streamlit as st

import time

from agent.core import get_memory,build_agent
from langchain.callbacks.streamlit import StreamlitCallbackHandler

#Page Config

st.set_page_config(
    page_title="AI Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


#Session State  Bootstrap
#Runs once per browser session. Preserves history across reruns

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thinking" not in st.session_state:
    st.session_state.thinking = False
    

# Sidebar

with st.sidebar:
    st.title("AI Agent")
    st.caption("Multi-agent · RAG · Web search")
    
    st.divider()
    
    #Model selector placeholder
    
    model = st.selectbox(
        "Model",
        ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"],
    )
    
    #st.slider(label, min_value, max_value, value, step)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    
    st.divider()
    
    if st.button("Clear Chat",use_container_width=True):
        st.session_state.messages = []
        st.session_state.memory=get_memory() #reset memory
        st.rerun()
        
    st.divider()
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
        
        st_callback=StreamlitCallbackHandler(
            st.container(),
            expand_new_thoughts=True,
            collapse_completed_thoughts=True,
        )
        
        agent=build_agent(model,temperature)
        
        with st.spinner("Thinking..."):
            ressult=agent.invoke(
                {
                    "inputs": prompt,
                    "chat_history": st.session_state.memory.chat_memory.messages,
                    
                },
                callbacks=[st_callback],
            )
            
        response=ressult["output"]
        
        st.markdown(response)
        
        #Update memory with assistant response
        st.session_state.memory.chat_memory.add_user_message(prompt)
        st.session_state.memory.chat_memory.add_ai_message(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
        
        
        
        
        
    
    
    
    
