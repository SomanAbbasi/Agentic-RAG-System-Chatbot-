import uuid
import streamlit as st
from langchain_core.messages import HumanMessage, ChatMessage
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from agent.core import build_agent, get_memory
from agent.validator import Validator, build_retry_prompt
from agent.router import Router, RouteResult
import os

st.set_page_config(
    page_title="AI Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Session state
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

# Sidebar
with st.sidebar:
    st.title("AI Agent")
    st.caption("Multi-agent | RAG | Web search | Validator | Router")
    st.divider()

    agent_model = st.selectbox(
        "Agent model",
        [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
        ],
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    st.divider()

    use_validator = st.toggle("Enable validator", value=True)
    max_retries   = st.slider("Max retries", 1, 3, 1) if use_validator else 0
    st.divider()

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages      = []
        st.session_state.clean_history = []
        st.session_state.checkpointer  = get_memory()
        st.session_state.thread_id     = str(uuid.uuid4())
        st.rerun()

    st.divider()
    st.caption(f"Thread: `{st.session_state.thread_id[:8]}...`")
    st.caption("Phase 7 - Router active")


# Route intent badge labels and colours
ROUTE_META = {
    "chitchat": ("Chitchat",  "blue"),
    "factual":  ("Factual",   "orange"),
    "rag":      ("RAG",       "green"),
    "math":     ("Math",      "violet"),
}


# Build clean message history
def build_messages(user_prompt: str) -> list:
    """
    Builds message list from clean history + new prompt.
    Uses ChatMessage for assistant turns — avoids tool_calls=[]
    which causes Groq to fail on subsequent tool-calling turns.
    """
    messages = []
    for turn in st.session_state.clean_history:
        messages.append(HumanMessage(content=turn["user"]))
        messages.append(
            ChatMessage(content=turn["assistant"], role="assistant")
        )
    messages.append(HumanMessage(content=user_prompt))
    return messages


def handle_chitchat(prompt: str) -> str:
    """
    Direct LLM call for chitchat — skips agent, tools, and validator.
    Uses 8b model since no reasoning or tool use needed.
    One LLM call, fast, cheap.
    """
    try:
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.8,
            api_key=os.getenv("GROQ_API_KEY"),
        )
        history_text = ""
        if st.session_state.clean_history:
            last = st.session_state.clean_history[-1]
            history_text = (
                f"Previous exchange: "
                f"User said '{last['user']}', "
                f"you replied '{last['assistant']}'. "
            )
        response = llm.invoke([
            SystemMessage(content=(
                "You are a friendly, helpful AI assistant. "
                "Respond naturally to casual conversation. "
                f"{history_text}"
                "Keep responses concise and warm."
            )),
            HumanMessage(content=prompt),
        ])
        return response.content
    except Exception as e:
        return f"Hi! How can I help you today?"


# direct calculator, no full agent needed
def handle_math(prompt: str) -> str:
   
    try:
        import re, numexpr as ne
        # Try to extract a math expression from the prompt
        expr = re.sub(r'[^0-9+\-*/().** ]', '', prompt).strip()
        if expr:
            result = ne.evaluate(expr)
            return f"The result is **{result}**"
    except Exception:
        pass
    return None


# Main agent runner
def run_agent(prompt: str, model: str, temperature: float):
    """
    Runs full agent pipeline with clean message history.
    Returns (response, tool_messages, error).
    Retries once with streaming=False on tool call failures.
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
        return *_invoke(use_streaming=True), None
    except Exception as e:
        err = str(e)
        if "Failed to call a function" in err:
            try:
                return *_invoke(use_streaming=False), None
            except Exception as e2:
                return None, [], str(e2)
        return None, [], err


# Validator runner
def run_validator(validator, question: str, response: str):
    try:
        return validator.validate(question, response)
    except Exception:
        return None


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

        router     = st.session_state.router
        validator  = st.session_state.validator

        final_response  = None
        final_tool_msgs = []
        validation      = None
        route           = None
        attempt         = 0
        error_msg       = None

        with st.spinner("Thinking..."):

            #  classify intent
            route = router.classify(prompt)

            if route.intent == "chitchat":
                final_response = handle_chitchat(prompt)
                attempt = 1

            elif route.intent == "math":
                math_result = handle_math(prompt)
                if math_result:
                    final_response = math_result
                    attempt = 1
                else:
                    route.intent = "factual"  

            if route.intent in ("factual", "rag"):
                # pipeline ( agent + validator + retry loop)
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

        # Render
        if error_msg and not final_response:
            st.error("Something went wrong. Please try again.", icon="🚨")
            with st.expander("Error details", expanded=False):
                st.code(error_msg, language="text")

        else:
            st.markdown(final_response)

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
                    f" | {attempt-1} "
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
                        f"Could not validate after {max_retries} retries"
                        f" - Score: {validation.score}/10. "
                        "Showing best attempt.",
                        icon="⚠️",
                    )
            elif use_validator and validation is None and route and \
                    route.intent in ("factual", "rag"):
                st.info("Validator unavailable.", icon="ℹ️")

            # Tool calls
            if final_tool_msgs:
                with st.expander("Agent tool calls", expanded=False):
                    for m in final_tool_msgs:
                        st.markdown(f"**Tool:** `{m.name}`")
                        st.markdown(f"**Result:** {m.content}")

            # Validator trace
            if use_validator and validation:
                with st.expander("Validator trace", expanded=False):
                    st.markdown(f"**Passed:** {validation.passed}")
                    st.markdown(f"**Score:** {validation.score}/10")
                    st.markdown(f"**Reason:** {validation.reason}")
                    st.markdown(f"**Suggestion:** {validation.suggestion}")
                    st.markdown(f"**Attempts:** {attempt}")

            # Router trace
            if route:
                with st.expander("Router trace", expanded=False):
                    st.markdown(f"**Intent:** {route.intent}")
                    st.markdown(f"**Confidence:** {route.confidence}/10")
                    st.markdown(f"**Reason:** {route.reason}")

        # Save to clean history
        if final_response:
            st.session_state.clean_history.append({
                "user": prompt,
                "assistant": final_response,
            })
            st.session_state.messages.append(
                {"role": "assistant", "content": final_response}
            )