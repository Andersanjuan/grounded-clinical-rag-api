from typing import Dict, Any, List
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from app.rag.retriever import retrieve_context
from app.config import settings

from app.state import state


SYSTEM_PROMPT = """You are a clinical QA assistant.
You must answer using ONLY the provided context.

If the answer is not explicitly supported by the context, say: "I don't know based on the provided documents."
You must include citations in the format [source_id], where source_id matches the chunk_id shown in the context.
Do not cite anything you did not use.

"""

USER_PROMPT = """Question:
{question}

Context (each chunk has a chunk_id):
{context}

Write a concise answer with citations.
"""

def should_abstain(chunks, max_distance: float = 0.8) -> bool:
    # Conservative default for small demo corpus; we can tune later
    distances = [c.distance for c in chunks if c.distance is not None]
    if not distances:
        return True
    best = min(distances)
    return best > max_distance

def best_distance(chunks) -> float | None:
    distances = [c.distance for c in chunks if c.distance is not None]
    return min(distances) if distances else None

def format_context(chunks) -> str:
    # Make the context explicit and citation-friendly
    lines: List[str] = []
    for c in chunks:
        lines.append(f"chunk_id: {c.chunk_id}\ntext: {c.text}")
    return "\n\n---\n\n".join(lines)


def answer_question(question: str, top_k: int = 3) -> Dict[str, Any]:
    chunks, citations = retrieve_context(question, top_k=top_k)

    # --- grounding diagnostics ---
    distances = [c.distance for c in chunks if c.distance is not None]
    best_distance = min(distances) if distances else None

    abstain = (
        best_distance is None
        or best_distance > settings.max_distance
    )

    grounding = {
        "best_distance": best_distance,
        "max_distance_threshold": settings.max_distance,
        "abstained": abstain,
    }

    if abstain:
        return {
            "question": question,
            "answer": "I don't know based on the provided documents.",
            "citations": citations,
            "chunks": [c.model_dump() for c in chunks],
            "warning_flags": ["low_retrieval_confidence"],
            "grounding": grounding,
        }

    context_str = format_context(chunks)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("user", USER_PROMPT),
        ]
    )

    if state.llm is None:
        raise RuntimeError("LLM not initialized. Startup hook did not run.")

    chain = prompt | state.llm
    response = chain.invoke({"question": question, "context": context_str})
    answer_text = response.content

    if not any(cid in answer_text for cid in citations):
        return {
            "question": question,
            "answer": "I don't know based on the provided documents.",
            "citations": citations,
            "chunks": [c.model_dump() for c in chunks],
            "warning_flags": ["missing_citations_in_answer"],
            "grounding": grounding,
        }

    return {
        "question": question,
        "answer": answer_text,
        "citations": citations,
        "chunks": [c.model_dump() for c in chunks],
        "warning_flags": [],
        "grounding": grounding,
    }

