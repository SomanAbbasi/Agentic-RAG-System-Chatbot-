import streamlit as st

import time

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
        help="Groq model selection — active from Phase 3",
    )
    
    #st.slider(label, min_value, max_value, value, step)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    
    st.divider()
    
    st.caption("Phase 2 — UI shell")
    
#Main Chat Area

st.title("AI Agent")


# Render all existing messages from session state
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        

#streaming function

def mock_stream(user_input: str):
    """
    Simulates a streaming LLM response word by word.
    TODO// replace this with real Groq streaming in Phase 3.
    """
    response = (
        f"[Mock response] You said: '{user_input}'. "
        "In Phase 3 this becomes a real Groq LLM call "
        "running inside a LangChain ReAct agent loop."
    )
    for word in response.split():
        yield word + " "
        time.sleep(0.05)
    
    
#Chat input handler 

if prompt := st.chat_input("Type your message here..."):
    
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    #Stream assistant response
    with st.chat_message("assistant"):
        
        response=st.write_stream(mock_stream(prompt))
    
    
    # Save completed response to history
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    
