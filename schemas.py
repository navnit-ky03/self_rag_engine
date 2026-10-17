from typing import TypedDict, Literal, Optional
from pydantic import BaseModel, Field
from langchain_core.documents import Document

class State(TypedDict):
    question: str
    need_retrival: bool
    
    retrieval_query: str
    retrieval_tries: int
    
    docs: list[Document]
    answer: str    
    
    relevant_docs: list[Document]
    context: str
    
    issup: Literal["fully_supported", "partially_supported", "no_support"]
    evidence: list[str]
    
    retries: int
    
    isuse: Literal["useful", "not_useful"]
    reason: str

    chat_history: list[dict]  # Accumulates conversation turns for session memory

class ChatRequest(BaseModel):
    question: str
    thread_id: Optional[str] = "default_session"

class ChatResponse(BaseModel):
    answer: str

class DecisionState(BaseModel):
    need_retrival: bool = Field(..., description="Use this schema to answer in True or False for answering user query, True for needing Retrival else False ")
    
class relevantstate(BaseModel):
    relevant: bool = Field(
        ...,
        description="True if the document helps answer the question, else False."
    )
    
class issup(BaseModel):
    issup: Literal["fully_supported", "partially_supported", "no_support"]
    evidence: list[str] = Field(default_factory=list)

class isUse(BaseModel):
    isuse: Literal["useful", "not_useful"]
    reason: str = Field(..., description="short 1 line reason")
    
class retrieval(BaseModel):
    retrieval_query: str = Field(..., description="Rewritten query optimized for vector retrieval against internal company PDFs.")