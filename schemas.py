
from typing import TypedDict, Literal, Optional
from pydantic import BaseModel, Field,field_validator
from langchain_core.documents import Document


class State(TypedDict, total=False):
    question: str
    need_retrival: bool
    rewrite_unchanged: bool
    retrieval_query: str
    retrieval_tries: int

    docs: list[Document]
    relevant_docs: list[Document]
    context: str
    answer: str

    issup: Literal[
        "fully_supported",
        "partially_supported",
        "no_support",
    ]
    evidence: list[str]
    retries: int

    isuse: Literal["useful", "not_useful"]
    reason: str

    chat_history: list[dict]


class ChatRequest(BaseModel):
    question: str
    thread_id:str = Field(...,min_length = 1,max_length = 128)
    @field_validator("thread_id")
    @classmethod
    def validate_thread_id(cls,value:str)->str:
        value = value.strip()
        if not value:
            raise ValueError("Provide Valid Thread id")
        return value


class ChatResponse(BaseModel):
    answer: str


class DecisionState(BaseModel):
    need_retrival: bool = Field(
        ...,
        description=(
            "True when the question requires facts from "
            "Navnit's portfolio knowledge base. "
            "False for greetings and general conversation."
        ),
    )


class relevantstate(BaseModel):
    relevant: bool = Field(
        ...,
        description=(
            "True if the retrieved document contains "
            "information relevant to answering the question."
        ),
    )

class BatchRelevanceState(BaseModel):
    relevant: list[bool] = Field(
        ...,
        description=(
            "A list of relevance decisions for retrieved documents. "
            "Each boolean corresponds to a document in the same order. "
            "True means relevant, False means irrelevant."
        )
    )

class issup(BaseModel):
    issup: Literal[
        "fully_supported",
        "partially_supported",
        "no_support",
    ]
    evidence: list[str] = Field(default_factory=list)


class isUse(BaseModel):
    isuse: Literal["useful", "not_useful"]
    reason: str = Field(
        ...,
        description=(
            "A factual, appropriately qualified answer is useful "
            "even when the knowledge base does not document "
            "the requested information. Do not penalize honest "
            "uncertainty or unsupported-detail refusal."
        ),
    )


class retrieval(BaseModel):
    retrieval_query: str = Field(
        ...,
        min_length=1,
        description=(
            "Concise search query optimized for semantic retrieval "
            "from Navnit Kumar Yadav's portfolio knowledge base. "
            "Preserve the user's original intent and do not "
            "invent facts."
        ),
    )
