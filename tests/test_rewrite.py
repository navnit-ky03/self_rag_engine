
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from config import GROQ_MODEL
from schemas import retrieval

load_dotenv()

llm = ChatGroq(
    model_name=GROQ_MODEL,
    temperature=0.2,
)

rewrite_llm = llm.with_structured_output(
    retrieval,
    method="json_mode",
)

try:
    result = rewrite_llm.invoke(
        [
            (
                "system",
                "Return only a valid JSON object with exactly "
                "one string field named retrieval_query."
            ),
            (
                "human",
                "Rewrite this for semantic search: "
                "How many years of professional experience "
                "does Navnit have, and where has he worked?"
            ),
        ]
    )

    print("Result:", result.model_dump())
    print("Groq structured output test passed.")

except Exception as exc:
    print("Groq structured output test failed:")
    print(type(exc).__name__, str(exc))
