import uuid
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ChatMessage, SystemMessage
from langchain_groq import ChatGroq
from agent.core import build_agent, get_memory
from agent.validator import Validator, build_retry_prompt
from agent.router import Router
from rag.ingestor import ingest_bytes, get_ingested_files

load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Agent Builder",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed", # Hide sidebar as we use columns
)

# --- CSS Injection ---
st.markdown("""
<style>
    /* Global Background & Font */
    [data-testid="stAppViewContainer"] {
        background-color: #0B0D11;
        font-family: 'IBM Plex Mono', monospace;
        color: #FFFFFF;
    }
    
    /* Typography */
    h4 {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 10px !important;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #5A6480;
        margin-top: 2rem !important;
        margin-bottom: 0.5rem !important;
    }

    /* Selectbox Styling */
    div[data-baseweb="select"] {
        background-color: #141720 !important;
        border: 1px solid #242836 !important;
        border-radius: 0px !important;
    }
    div[data-baseweb="select"] > div {
        background-color: #141720 !important;
        color: #FFFFFF !important;
        font-family: 'IBM Plex Mono', monospace !important;
    }
    
    /* Slider Styling */
    div[data-testid="stSlider"] [data-testid="stThumb"] {
        background-color: #4F7CFF !important;
        border-radius: 0px !important;
        width: 12px !important;
        height: 12px !important;
    }
    div[data-testid="stSlider"] [data-testid="stSliderTickBar"] {
        background-color: #242836 !important;
    }

    /* Toggle Styling */
    div[data-testid="stCheckbox"] label span {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 12px;
    }
    /* We target the switch track for st.toggle (which is actually a checkbox in Streamlit) */
    .st-at {
        background-color: #4F7CFF !important;
    }

    /* Chat Input Styling */
    div[data-testid="stChatInput"] {
        border-top: 1px solid #242836 !important;
        background-color: #0B0D11 !important;
        padding-left: 0px !important;
        padding-right: 0px !important;
    }
    div[data-testid="stChatInput"] textarea {
        background-color: transparent !important;
        border: none !important;
        color: #FFFFFF !important;
        font-family: 'Geist Mono', monospace !important;
    }

    /* Chat Messages Styling */
    [data-testid="stChatMessage"] {
        background-color: transparent !important;
        border: none !important;
        padding-left: 0px !important;
    }
    [data-testid="stChatMessageContent"] {
        font-family: 'Geist Mono', monospace !important;
        font-size: 14px;
        color: #E0E0E0;
    }
    /* User message specific */
    [data-testid="stChatMessage"][data-testid="user"] {
        background-color: #141720 !important;
        border-radius: 0px !important;
    }

    /* Status Bar */
    .status-bar {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 10px;
        color: #5A6480;
        margin-bottom: 2rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    /* Pipeline Schematic */
    .pipeline-schematic {
        font-family: 'IBM Plex Mono', monospace;
        color: #242836;
        text-align: center;
        margin-top: 20vh;
        font-size: 14px;
        letter-spacing: 0.2em;
    }
    
    /* Custom buttons (Deploy, Clear) */
    .stButton>button {
        border-radius: 0px !important;
        font-family: 'IBM Plex Mono', monospace !important;
        text-transform: uppercase !important;
        letter-spacing: 0.1em !important;
        font-size: 10px !important;
    }
    .deploy-btn button {
        border: 1px solid #4F7CFF !important;
        background-color: transparent !important;
        color: #4F7CFF !important;
    }
    
    /* Sidebar column */
    .sidebar-col {
        background-color: #141720;
        padding: 2rem;
        height: 100vh;
        border-right: 1px solid #242836;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State ---
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "checkpointer" not in st.session_state:
    st.session_state.checkpointer = get_memory()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "clean_history" not in st.session_state:
    st.session_state.clean_history = []
if "validator" not in st.session_state:
    st.session_state.validator = Validator()
if "router" not in st.session_state:
    st.session_state.router = Router()
if "token_count" not in st.session_state:
    st.session_state.token_count = 0
if "message_count" not in st.session_state:
    st.session_state.message_count = 0
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = set()

# --- Helper Functions ---
ROUTE_META = {
    "chitchat": ("Chitchat", "blue"),
    "factual":  ("Factual",  "orange"),
    "rag":      ("RAG",      "green"),
    "math":     ("Math",     "violet"),
}

def estimate_tokens(text: str) -> int:
    return len(text) // 4

def build_messages(user_prompt: str) -> list:
    messages = []
    for turn in st.session_state.clean_history:
        messages.append(HumanMessage(content=turn["user"]))
        messages.append(ChatMessage(content=turn["assistant"], role="assistant"))
    messages.append(HumanMessage(content=user_prompt))
    return messages

def handle_chitchat(prompt: str) -> str:
    try:
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.8,
            api_key=os.getenv("GROQ_API_KEY"),
            streaming=False,
        )
        history_text = ""
        if st.session_state.clean_history:
            last = st.session_state.clean_history[-1]
            history_text = f"Previous: user said '{last['user']}', you replied '{last['assistant']}'. "
        response = llm.invoke([
            SystemMessage(content=f"You are a friendly helpful AI assistant. Respond naturally to casual conversation. {history_text} Keep responses concise and warm."),
            HumanMessage(content=prompt),
        ])
        return response.content
    except Exception:
        return "Hello! How can I help you today?"

def handle_math(prompt: str):
    try:
        import re
        import numexpr as ne
        expr = re.sub(r'[^0-9+\-*/().** ]', '', prompt).strip()
        if expr and len(expr) > 1:
            result = ne.evaluate(expr)
            return f"The result is **{result}**"
    except Exception:
        pass
    return None

def run_agent(prompt: str, model: str, temperature: float):
    def _invoke(use_streaming: bool):
        agent = build_agent(
            model=model,
            temperature=temperature,
            checkpointer=st.session_state.checkpointer,
            streaming=use_streaming,
        )
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        result = agent.invoke({"messages": build_messages(prompt)}, config=config)
        response = result["messages"][-1].content
        tool_msgs = [m for m in result["messages"] if hasattr(m, "type") and m.type == "tool"]
        return response, tool_msgs

    try:
        return *_invoke(True), None
    except Exception as e:
        err = str(e)
        if "Failed to call a function" in err:
            try:
                return *_invoke(False), None
            except Exception as e2:
                return None, [], str(e2)
        return None, [], err

def run_validator(validator, question: str, response: str):
    try:
        return validator.validate(question, response)
    except Exception:
        return None

def stream_text(text: str):
    import time
    for word in text.split():
        yield word + " "
        time.sleep(0.03)

# --- Main Layout ---
col_settings, col_chat = st.columns([1, 3])

# --- Settings Column (Left) ---
with col_settings:
    st.markdown("#### MODEL SETTINGS")
    agent_model = st.selectbox(
        "Agent model",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
        label_visibility="collapsed"
    )
    st.markdown("#### TEMPERATURE")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1, label_visibility="collapsed")
    
    st.markdown("#### VALIDATOR")
    use_validator = st.toggle("Enable validator", value=True)
    max_retries = st.slider("Max retries", 1, 3, 1) if use_validator else 0
    
    st.markdown("#### KNOWLEDGE BASE")
    uploaded = st.file_uploader(
        "Upload PDF or TXT",
        type=["pdf", "txt"],
        label_visibility="collapsed"
    )

    if uploaded is not None:
        file_key = f"{uploaded.name}_{uploaded.size}"
        if file_key not in st.session_state.ingested_files:
            with st.spinner(f"Ingesting..."):
                try:
                    file_bytes = uploaded.read()
                    if len(file_bytes) > 0:
                        count = ingest_bytes(file_bytes, uploaded.name)
                        st.session_state.ingested_files.add(file_key)
                        st.success(f"Stored {count} chunks", icon="✅")
                except Exception as e:
                    st.error(f"Ingestion failed", icon="🚨")

    st.markdown("#### SESSION STATS")
    st.metric("Messages", st.session_state.message_count)
    st.metric("Est. tokens", f"{st.session_state.token_count:,}")

    if st.button("Clear session", use_container_width=True):
        st.session_state.messages = []
        st.session_state.clean_history = []
        st.session_state.checkpointer = get_memory()
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.token_count = 0
        st.session_state.message_count = 0
        st.rerun()

# --- Chat Column (Right) ---
with col_chat:
    # Top bar with status and deploy button
    stat_col, btn_col = st.columns([4, 1])
    with stat_col:
        st.markdown(f"""
        <div class='status-bar'>
          AI AGENT &nbsp;·&nbsp; {agent_model} &nbsp;·&nbsp; t={temperature:.2f} &nbsp;▌
        </div>
        """, unsafe_allow_html=True)
    with btn_col:
        st.markdown('<div class="deploy-btn">', unsafe_allow_html=True)
        st.button("[ DEPLOY ]", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Empty State Pipeline Schematic
    if not st.session_state.messages:
        st.markdown("""
        <div class='pipeline-schematic'>
          [ROUTER] ──► [RAG] ──► [VALIDATOR] ──► [RESPONSE]
        </div>
        """, unsafe_allow_html=True)

    # Message Display
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input Handler
    if prompt := st.chat_input("Ask anything..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            router = st.session_state.router
            validator = st.session_state.validator
            final_response = None
            final_tool_msgs = []
            validation = None
            route = None
            attempt = 0
            error_msg = None

            with st.spinner("Routing..."):
                route = router.classify(prompt)

            if route.intent == "chitchat":
                with st.spinner("Responding..."):
                    text = handle_chitchat(prompt)
                final_response = st.write_stream(stream_text(text))
                attempt = 1
            elif route.intent == "math":
                math_result = handle_math(prompt)
                if math_result:
                    final_response = math_result
                    st.markdown(final_response)
                    attempt = 1
                else:
                    route.intent = "factual"

            if route.intent in ("factual", "rag"):
                with st.spinner("Thinking..."):
                    response, tool_msgs, error = run_agent(prompt, agent_model, temperature)
                    attempt = 1
                    if error:
                        error_msg = error
                    else:
                        final_response = response
                        final_tool_msgs = tool_msgs
                        if use_validator:
                            validation = run_validator(validator, prompt, response)
                            while validation is not None and not validation.passed and attempt <= max_retries:
                                retry_prompt = build_retry_prompt(prompt, response, validation, attempt)
                                response, tool_msgs, error = run_agent(retry_prompt, agent_model, temperature)
                                attempt += 1
                                if error: break
                                final_response = response
                                final_tool_msgs = tool_msgs
                                validation = run_validator(validator, prompt, response)

                if not error_msg and final_response:
                    st.markdown(final_response)

            # Execution Results & Badges
            if error_msg and not final_response:
                st.error("Error encountered.", icon="🚨")
            else:
                if route:
                    label, colour = ROUTE_META.get(route.intent, ("Unknown", "grey"))
                    st.caption(f"Route: {label} | Confidence: {route.confidence}/10")
                if use_validator and validation:
                    if validation.passed:
                        st.caption(f"Validated: {validation.score}/10")
                    else:
                        st.caption(f"Validation Failed: {validation.score}/10")

            # Save to History
            if final_response:
                st.session_state.token_count += estimate_tokens(prompt + final_response)
                st.session_state.message_count += 1
                st.session_state.clean_history.append({"user": prompt, "assistant": final_response})
                st.session_state.messages.append({"role": "assistant", "content": final_response})
