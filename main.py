import uuid
import streamlit as st
from langchain.messages import HumanMessage
from agent.core import build_agent, get_memory
from agent.validator import Validator, build_retry_prompt

st.set_page_config(
    page_title="AI Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─ Session state bootstrap ──────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "checkpointer" not in st.session_state:
    st.session_state.checkpointer = get_memory()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "validator" not in st.session_state:
    st.session_state.validator = Validator()

# ─ Sidebar 
with st.sidebar:
    st.title("AI Agent")
    st.caption("Multi-agent · RAG · Web search · Validator")
    st.divider()

    model = st.selectbox(
        "Model",
        ["llama-3.1-8b-instant","llama-3.3-70b-versatile", "mixtral-8x7b-32768"],
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)

    st.divider()

    use_validator = st.toggle("Enable validator", value=True)
    max_retries   = st.slider("Max retries", 1, 3, 2) if use_validator else 0

    st.divider()

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.checkpointer = get_memory()
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

    st.divider()
    st.caption(f"Thread: `{st.session_state.thread_id[:8]}...`")
    st.caption("Phase 6 — Validator active")


def run_agent(prompt: str, model: str, temperature: float):
    """
    Runs the agent once. Returns (response, tool_messages, error).
    Never raises — caller always gets a clean tuple back.
    """
    try:
        agent = build_agent(
            model=model,
            temperature=temperature,
            checkpointer=st.session_state.checkpointer,
        )
        config = {"configurable": {"thread_id": st.session_state.thread_id}}

        result = agent.invoke(
            {"messages": [HumanMessage(content=prompt)]},
            config=config,
        )

        response = result["messages"][-1].content
        tool_msgs = [
            m for m in result["messages"]
            if hasattr(m, "type") and m.type == "tool"
        ]
        return response, tool_msgs, None

    except Exception as e:
        return None, [], str(e)


def run_validator(validator: Validator, question: str, response: str):

    try:
        return validator.validate(question, response)
    except Exception:
        return None


#  Main chat area 
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

        validator      = st.session_state.validator
        final_response = None
        final_tool_msgs = []
        validation     = None
        attempt        = 0
        error_msg      = None

        with st.spinner("Thinking..."):

            response, tool_msgs, error = run_agent(prompt, model, temperature)
            attempt = 1

            if error:
                error_msg = error
            else:
                final_response  = response
                final_tool_msgs = tool_msgs

                if use_validator:
                    validation = run_validator(validator, prompt, response)

                    #  Retry loop
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
                            retry_prompt, model, temperature
                        )
                        attempt += 1

                        if error:
                            # Retry itself errored — stop loop,
                            # keep last good response
                            break

                        final_response  = response
                        final_tool_msgs = tool_msgs
                        validation = run_validator(validator, prompt, response)

        if error_msg and not final_response:
           
            st.error(
                "Something went wrong while generating a response. "
                "Please try again or rephrase your question.",
                icon="🚨",
            )
            with st.expander("Error details", expanded=False):
                st.code(error_msg, language="text")

        else:
            st.markdown(final_response)

            # Validation badge
            if use_validator and validation:
                retry_label = (
                    f" · {attempt - 1} retr{'y' if attempt == 2 else 'ies'}"
                    if attempt > 1 else ""
                )
                if validation.passed:
                    st.success(
                        f"Validated — Score: {validation.score}/10{retry_label}",
                        icon="✅",
                    )
                else:
                    st.warning(
                        f"Could not fully validate after {max_retries} "
                        f"retries — Score: {validation.score}/10. "
                        "Showing best attempt.",
                        icon="⚠️",
                    )
            elif use_validator and validation is None:
                st.info("Validator unavailable for this response.", icon="ℹ️")

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
                    st.markdown(f"**Attempts used:** {attempt}")

        # Save to history only if we got a real response
        if final_response:
            st.session_state.messages.append(
                {"role": "assistant", "content": final_response}
            )