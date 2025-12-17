from fastapi import APIRouter, Depends
from app.models.schemas import RetrieveRequest, RetrieveResponse
from app.rag.retriever import retrieve_context
from app.models.schemas import QueryRequest, QueryResponse
from app.rag.grounded_qa import answer_question

from app.security import require_api_key

import logging

router = APIRouter(prefix="", tags=["rag"])
logger = logging.getLogger("medrag")

@router.post("/retrieve", response_model=RetrieveResponse, dependencies=[Depends(require_api_key)])
def retrieve(req: RetrieveRequest):
    chunks, citations = retrieve_context(req.question, top_k=req.top_k)
    return RetrieveResponse(
        question=req.question,
        top_k=req.top_k,
        chunks=chunks,
        citations=citations,
    )

@router.post("/query", response_model=QueryResponse, dependencies=[Depends(require_api_key)])
def query(req: QueryRequest):
    result = answer_question(req.question, top_k=req.top_k)

    grounding = result.get("grounding", {})

    logger.info(
        "query_result status=ok abstained=%s best_distance=%s flags=%s",
        grounding.get("abstained"),
        grounding.get("best_distance"),
        result.get("warning_flags", []),
    )

    return QueryResponse(
        question=result["question"],
        answer=result["answer"],
        citations=result["citations"],
        chunks=result["chunks"],
        warning_flags=result.get("warning_flags", []),
        grounding=grounding,
    )

