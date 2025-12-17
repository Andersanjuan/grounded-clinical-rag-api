from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class RetrieveRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question")
    top_k: int = Field(3, ge=1, le=10, description="Number of chunks to retrieve")


class RetrievedChunk(BaseModel):
    rank: int
    chunk_id: str
    source_file: str
    metadata: Dict[str, Any]
    text: str
    distance: float | None = None

class RetrieveResponse(BaseModel):
    question: str
    top_k: int
    chunks: List[RetrievedChunk]
    citations: List[str]

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(3, ge=1, le=10)

class GroundingInfo(BaseModel):
    best_distance: float | None = None
    max_distance_threshold: float
    abstained: bool

class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: List[str]
    chunks: List[RetrievedChunk]
    warning_flags: List[str]
    grounding: GroundingInfo


