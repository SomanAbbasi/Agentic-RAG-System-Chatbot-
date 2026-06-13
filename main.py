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

st.set_page_config(
    page_title="AI Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

# Sidebar 
with st.sidebar:
    st.title("AI Agent")
    st.caption("Multi-agent | RAG | Web search | Validator | Router")
    st.divider()

    st.subheader("Model settings")
    agent_model = st.selectbox(
        "Agent model",
        [
            "llama-3.3-70b-versatile",   # reliable tool calling
            "llama-3.1-8b-instant",      # faster but less reliable
        ],
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    st.divider()

    st.subheader("Validator")
    use_validator = st.toggle("Enable validator", value=True)
    max_retries   = st.slider("Max retries", 1, 3, 1) if use_validator else 0
    st.divider()

    #  Document upload 
    st.subheader("Knowledge base")
    uploaded = st.file_uploader(
        "Upload PDF or TXT",
        type=["pdf", "txt"],
        help="Embedded and stored locally in ChromaDB",
    )

    if uploaded is not None:
        file_key = f"{uploaded.name}_{uploaded.size}"

        if file_key not in st.session_state.ingested_files:
            with st.spinner(f"Ingesting {uploaded.name}..."):
                try:
                    file_bytes = uploaded.read()

                    # Validate bytes before ingesting
                    if len(file_bytes) == 0:
                        st.error(
                            "File appears empty. Please re-upload.",
                            icon="🚨",
                        )
                    else:
                        count = ingest_bytes(file_bytes, uploaded.name)
                        # Mark as ingested so reruns skip it
                        st.session_state.ingested_files.add(file_key)
                        st.success(
                            f"Stored {count} chunks from {uploaded.name}",
                            icon="✅",
                        )
                except Exception as e:
                    st.error(f"Ingestion failed: {e}", icon="🚨")
        else:
            st.info(
                f"{uploaded.name} already ingested this session.",
                icon="ℹ️",
            )

    ingested = get_ingested_files()
    if ingested:
        with st.expander(
            f"Ingested files ({len(ingested)})", expanded=False
        ):
            for f in ingested:
                st.caption(f"- {f}")
    else:
        st.caption("No files ingested yet.")

    st.divider()

    st.subheader("Session stats")
    col1, col2 = st.columns(2)
    col1.metric("Messages", st.session_state.message_count)
    col2.metric("Est. tokens", f"{st.session_state.token_count:,}")
    st.divider()

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages      = []
        st.session_state.clean_history = []
        st.session_state.checkpointer  = get_memory()
        st.session_state.thread_id     = str(uuid.uuid4())
        st.session_state.token_count   = 0
        st.session_state.message_count = 0
        # Note: ingested_files not cleared — RAG persists across chat resets
        st.rerun()

    st.divider()
    st.caption(f"Thread: `{st.session_state.thread_id[:8]}...`")
    st.caption("Phase 8 - Final")


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
        messages.append(
            ChatMessage(content=turn["assistant"], role="assistant")
        )
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
            history_text = (
                f"Previous: user said '{last['user']}', "
                f"you replied '{last['assistant']}'. "
            )
        response = llm.invoke([
            SystemMessage(content=(
                "You are a friendly helpful AI assistant. "
                "Respond naturally to casual conversation. "
                f"{history_text}"
                "Keep responses concise and warm."
            )),
            HumanMessage(content=prompt),
        ])
        return response.content
    except Exception:
        return "Hello! How can I help you today?"


def handle_math(prompt: str):
    """
    Tries to evaluate math directly with numexpr.
    Returns result string on success, None on failure.
    Caller falls back to full agent if None returned.
    """
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
    """
    Runs the full agent pipeline once.
    Returns (response, tool_messages, error) — never raises.

    On 'Failed to call a function' error from Groq (happens when
    model generates malformed tool call JSON with streaming on),
    retries once with streaming=False which is more stable.
    """
    def _invoke(use_streaming: bool):
        agent = build_agent(
            model=model,
            temperature=temperature,
            checkpointer=st.session_state.checkpointer,
            streaming=use_streaming,
        )
        
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}

        result = agent.invoke(
            {"messages": build_messages(prompt)},
            config=config,
        )
        response = result["messages"][-1].content
        tool_msgs = [
            m for m in result["messages"]
            if hasattr(m, "type") and m.type == "tool"
        ]
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
    """Word-by-word generator for streaming chitchat responses."""
    import time
    for word in text.split():
        yield word + " "
        time.sleep(0.03)


# Main chat area 
st.title("AI Agent")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input handler 
if prompt := st.chat_input("Ask anything..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        router    = st.session_state.router
        validator = st.session_state.validator

        final_response  = None
        final_tool_msgs = []
        validation      = None
        route           = None
        attempt         = 0
        error_msg       = None

        #  classify intent 
        with st.spinner("Routing..."):
            route = router.classify(prompt)

        #  route to pipeline

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
            # ( agent + optional validator + retry loop )
            with st.spinner("Thinking..."):
                response, tool_msgs, error = run_agent(
                    prompt, agent_model, temperature
                )
                attempt = 1

                if error:
                    error_msg = error
                else:
                    final_response  = response
                    final_tool_msgs = tool_msgs

                    if use_validator:
                        validation = run_validator(
                            validator, prompt, response
                        )

                        while (
                            validation is not None
                            and not validation.passed
                            and attempt <= max_retries
                        ):
                            retry_prompt = build_retry_prompt(
                                original_question=prompt,
                                failed_response=response,
                                validation=validation,
                                attempt=attempt,
                            )
                            response, tool_msgs, error = run_agent(
                                retry_prompt, agent_model, temperature
                            )
                            attempt += 1

                            if error:
                                break

                            final_response  = response
                            final_tool_msgs = tool_msgs
                            validation = run_validator(
                                validator, prompt, response
                            )

            if not error_msg and final_response:
                st.markdown(final_response)

        #  render results 

        if error_msg and not final_response:
            st.error(
                "Something went wrong. Please try again.",
                icon="🚨",
            )
            with st.expander("Error details", expanded=False):
                st.code(error_msg, language="text")

        else:
            # Route badge
            if route:
                label, colour = ROUTE_META.get(
                    route.intent, ("Unknown", "grey")
                )
                st.markdown(
                    f":{colour}[Route: **{label}**"
                    f" | Confidence: {route.confidence}/10"
                    f" | {route.reason}]"
                )

            # Validator badge
            if use_validator and validation:
                retry_label = (
                    f" | {attempt - 1} "
                    f"retr{'y' if attempt == 2 else 'ies'}"
                    if attempt > 1 else ""
                )
                if validation.passed:
                    st.success(
                        f"Validated - Score: {validation.score}/10"
                        f"{retry_label}",
                        icon="✅",
                    )
                else:
                    st.warning(
                        f"Could not validate after {max_retries} "
                        f"retries - Score: {validation.score}/10. "
                        "Showing best attempt.",
                        icon="⚠️",
                    )
            elif (
                use_validator
                and validation is None
                and route
                and route.intent in ("factual", "rag")
            ):
                st.info("Validator unavailable.", icon="ℹ️")

            # Tool calls expander
            if final_tool_msgs:
                with st.expander("Agent tool calls", expanded=False):
                    for m in final_tool_msgs:
                        st.markdown(f"**Tool:** `{m.name}`")
                        st.markdown(f"**Result:** {m.content}")

            # Validator trace expander
            if use_validator and validation:
                with st.expander("Validator trace", expanded=False):
                    st.markdown(f"**Passed:** {validation.passed}")
                    st.markdown(f"**Score:** {validation.score}/10")
                    st.markdown(f"**Reason:** {validation.reason}")
                    st.markdown(f"**Suggestion:** {validation.suggestion}")
                    st.markdown(f"**Attempts:** {attempt}")

            # Router trace expander
            if route:
                with st.expander("Router trace", expanded=False):
                    st.markdown(f"**Intent:** {route.intent}")
                    st.markdown(f"**Confidence:** {route.confidence}/10")
                    st.markdown(f"**Reason:** {route.reason}")

        # save to history
        if final_response:
            st.session_state.token_count += estimate_tokens(
                prompt + final_response
            )
            st.session_state.message_count += 1
            st.session_state.clean_history.append({
                "user": prompt,
                "assistant": final_response,
            })
            st.session_state.messages.append(
                {"role": "assistant", "content": final_response}
            )