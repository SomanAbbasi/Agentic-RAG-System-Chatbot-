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
    """
    A second LLM call that reviews the agent's draft response.
    Uses a fast, cheap model (8b) since this is evaluation only.
    Returns a ValidationResult — caller decides what to do with it.
    """

    def __init__(self):
       
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.0,        
            api_key=os.getenv("GROQ_API_KEY"),
        ).with_structured_output(ValidationResult)

    def validate(
        self,
        user_question: str,
        agent_response: str,
    ) -> ValidationResult:
        """
        Evaluates the agent response against the original question.
        Returns a ValidationResult with pass/fail, score, and feedback.
        """
        messages = [
            SystemMessage(content="""
                    You are a strict quality evaluator for AI responses.
                    Evaluate the agent response against the user question.
                    Return a structured assessment.

                    SCORING CRITERIA:
                    - Does the response directly answer the question asked? (most important)
                    - Is the information accurate and not hallucinated?
                    - Is the response complete — not cut off or vague?
                    - Is the response well-structured and clear?
                    - Did the agent use appropriate tools when needed?

                    PASS CONDITIONS (passed=true, score >= 7):
                    - Response clearly and completely answers the question
                    - No obvious hallucinations or invented facts
                    - Not evasive or overly generic

                    FAIL CONDITIONS (passed=false, score < 7):
                    - Response does not answer the question
                    - Contains fabricated information
                    - Response is incomplete or cut off
                    - Agent said it couldn't find info but didn't try all tools
                    - Response is off-topic
                """),
            HumanMessage(content=f"""
                    User question: {user_question}

                    Agent response: {agent_response}

                    Evaluate this response strictly.
                    """),
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