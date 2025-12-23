**RAG Clinical Assistant API**

A grounded Retrieval-Augmented Generation (RAG) backend designed for clinical and biomedical text, emphasizing accuracy, abstention, citation enforcement, privacy-aware logging, and containerized deployment.

This project demonstrates how to safely serve LLM-based question answering over internal clinical documentation using modern NLP tooling, while minimizing hallucinations and enabling auditability.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
**1. High-Level Overview**

FastAPI backend that:

    Ingests unstructured clinical text (TXT/PDF/HTML-ready)

    Embeds and stores content in a vector database (ChromaDB)

    Retrieves evidence with distance-based confidence scoring

    Generates answers via an LLM only when evidence is sufficient

    Enforces grounding through abstention and citation validation

    Exposes secure, containerized APIs suitable for internal use

Out-of-scope or low-confidence queries result in a safe refusal, rather than speculative answers.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
**2. Architecture**

**Document Ingestion**
  └── Load → Chunk → Embed
           └── ChromaDB (vector store)

**Query Flow**
  User Question
     └── Embed query
          └── Vector similarity search (top-k + distances)
               └── Grounding gate
                    ├── Low confidence → abstain
                    └── High confidence → LLM answer
                             └── Citation validation
                                   └── API response

Key design principle:
The LLM is never allowed to answer unless the retrieved evidence meets a configurable confidence threshold.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
**3. Key Features**

Build RAG Pipelines

    Document ingestion with configurable chunking

    Sentence-transformer embeddings (local, no external API required)

    Vector similarity search using ChromaDB

    Modular retriever and formatter logic

Orchestrate LLM Logic

    LangChain prompt orchestration

    Context assembly from retrieved chunks

    Chat-based LLM invocation via Ollama (local LLaMA 3.x)

Optimize for Accuracy (Grounding)

    Distance-based confidence threshold (max_distance)

    Automatic abstention on low-confidence retrieval

    Mandatory citation presence for non-abstained answers

    Grounding diagnostics returned in API response:
        {
        "grounding": {
            "best_distance": 0.47,
            "max_distance_threshold": 1.2,
            "abstained": false
        }
    }

Develop Secure APIs

    FastAPI backend with OpenAPI schema

    API-key authentication via request headers

    Environment-based configuration (no secrets in code)

Privacy & Compliance Posture

    No logging of raw questions or document content

    Privacy-aware request logging (path, status, latency only)

    Clear separation between evaluation tooling and production API

    Designed for internal clinical decision support, not diagnosis

Deployment & Operations

    Fully containerized with Docker

    Externalized configuration via environment variables

    Compatible with local or hosted Ollama deployments
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
**4. API Endpoints**

GET /health

Health check endpoint (unauthenticated).

POST /retrieve

Returns retrieved document chunks and similarity distances without invoking the LLM.

POST /query

Returns a grounded answer or abstains, including:

    Answer text (if confident)

    Source citations

    Retrieved chunks

    Grounding diagnostics

    Warning flags

All non-health endpoints require X-API-Key if configured.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
**5. Quickstart**

Local Development
    pip install -r requirements.txt
    uvicorn app.main:app --reload

Optional authentication:
    export API_KEY="changeme"

docker build -t medrag-api .
    docker run --rm -p 8000:8000 \
    -e API_KEY="changeme" \
    -e OLLAMA_BASE_URL="http://host.docker.internal:11434" \
    medrag-api

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
**6. Configuration (Environment Variables)**

Variable    Description
API_KEY	    Optional API key for request authentication
OLLAMA_BASE_URL Ollama server endpoint
OLLAMA_MODEL	LLM model name (default: llama3.1)
MAX_DISTANCE	Retrieval confidence threshold
CHROMA_DIR  Vector DB persistence directory

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
**7. Grounding Evaluation**
Grounding behavior is evaluated using a separate HTTP-based evaluation harness that exercises the deployed API end-to-end.

Example Results:
    [
    {
        "test": "in_scope_hand_hygiene",
        "abstained": false,
        "best_distance": 0.47,
        "has_citation_in_answer": true
    },
    {
        "test": "out_of_scope_antibiotic",
        "abstained": true,
        "best_distance": 1.60,
        "warning_flags": ["low_retrieval_confidence"]
    }
    ]

Summary
    In-scope questions: answered with citations

    Out-of-scope questions: safely refused

    Abstention behavior matched expectations in all test cases

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
**8. Limitations & Future Work**

Add PDF and HTML ingestion pipelines

Integrate structured clinical ontologies (e.g., SNOMED, UMLS)

Role-based access control (RBAC)

Audit logging suitable for regulated environments

Offline embedding + model evaluation benchmarks

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
**9. Disclaimer**

This project is a technical demonstration of grounded RAG systems.
It is not a clinical decision-making tool and should not be used for patient care.
