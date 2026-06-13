import os
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()


# Route schema
class RouteResult(BaseModel):
    intent: str       # chitchat | factual | rag | math
    reason: str       # one line explanation
    confidence: int   # 1-10


class Router:
   
    def __init__(self):
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.0,
            api_key=os.getenv("GROQ_API_KEY"),
        ).with_structured_output(RouteResult)

    def classify(self, message: str) -> RouteResult:
        messages = [
            SystemMessage(content=(
                "You are an intent classifier. "
                "Classify the user message into exactly one intent.\n\n"
                "intents:\n"
                "- chitchat: greetings, thanks, bye, how are you, "
                "your name, casual talk, opinions about preferences\n"
                "- factual: questions about real people, companies, "
                "news, current events, history, science, definitions, "
                "how things work, coding questions, general knowledge\n"
                "- rag: questions explicitly about uploaded documents, "
                "files, PDFs, or the knowledge base\n"
                "- math: arithmetic, calculations, percentages, "
                "numeric expressions, unit conversions\n\n"
                "Reply with the intent, a one line reason, "
                "and confidence 1-10."
            )),
            HumanMessage(content=f"Message: {message}"),
        ]
        try:
            return self.llm.invoke(messages)
        except Exception:
           
            return RouteResult(
                intent="factual",
                reason="classification failed, defaulting to factual",
                confidence=5,
            )