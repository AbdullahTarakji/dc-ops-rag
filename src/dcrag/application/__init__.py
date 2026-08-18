"""Application layer: use cases orchestrating the domain ports.

Nothing in here knows that Qdrant, Ollama or FastAPI exist. That is the test: if a module in
this package imports a third-party client library, the dependency has leaked inwards and
the abstraction needs fixing.
"""

from dcrag.application.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from dcrag.application.ingest_service import IngestReport, IngestService
from dcrag.application.retrieval_service import (
    DEFAULT_RERANK_CANDIDATES,
    DEFAULT_TOP_K,
    RetrievalOutcome,
    RetrievalService,
)

__all__ = [
    "DEFAULT_RERANK_CANDIDATES",
    "DEFAULT_RRF_K",
    "DEFAULT_TOP_K",
    "IngestReport",
    "IngestService",
    "RetrievalOutcome",
    "RetrievalService",
    "reciprocal_rank_fusion",
]
