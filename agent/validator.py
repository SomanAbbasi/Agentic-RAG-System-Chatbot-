import os
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()



class ValidationResult(BaseModel):
    passed: bool          # True = send to user, False = retry
    score: int            # 1-10 quality score
    reason: str           # why it passed or failed
    suggestion: str       # what the agent should do differently on retry


class Validator:

    def __init__(self):
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            api_key=os.getenv("GROQ_API_KEY"),
        ).with_structured_output(ValidationResult)

    def validate(
        self,
        user_question: str,
        agent_response: str,
    ) -> ValidationResult:

        truncated = agent_response[:1500]
        if len(agent_response) > 1500:
            truncated += "... [truncated]"

        messages = [
            SystemMessage(content=(
                "You are a quality evaluator for AI responses.\n\n"
                "IMPORTANT EXEMPTIONS - always score 9/10 and pass=true for:\n"
                "- Greetings (hi, hello, hey, good morning, etc)\n"
                "- Chitchat (how are you, thanks, bye, what is your name)\n"
                "- Simple conversational exchanges with no factual content\n"
                "These do not need tools and should never fail validation.\n\n"
                "FOR ALL OTHER RESPONSES score on:\n"
                "- Does it directly answer the question?\n"
                "- No fabricated or hallucinated facts?\n"
                "- Used search tools for factual/real-world questions?\n\n"
                "PASS = score 7 or above. FAIL = score below 7.\n"
                "Be strict about hallucinated facts in factual responses.\n"
                "Be lenient about conversational responses."
            )),
            HumanMessage(content=(
                f"Question: {user_question[:300]}\n\n"
                f"Response: {truncated}\n\n"
                "Evaluate and return your assessment."
            )),
        ]

        return self.llm.invoke(messages)


#  Retry prompt builder 
def build_retry_prompt(
    original_question: str,
    failed_response: str,
    validation: ValidationResult,
    attempt: int,
) -> str:
    """
    Builds an enhanced prompt for the retry attempt.
    Injects the validator's feedback so the agent knows
    exactly what went wrong and what to do differently.
    """
    return f"""
        The user asked: {original_question}

        Your previous response (attempt {attempt}) was:
        {failed_response}

        Quality check FAILED (score: {validation.score}/10).
        Reason: {validation.reason}
        What to do differently: {validation.suggestion}

        Please try again. Address the failure reason directly.
        Use a different tool or approach if needed.
        Provide a complete, accurate answer this time.
"""