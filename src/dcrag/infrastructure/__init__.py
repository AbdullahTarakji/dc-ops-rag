"""Infrastructure layer: adapters implementing the domain ports.

Adapters may import anything they need — Qdrant, sentence-transformers, httpx — and are the
only place allowed to. None of them inherit from a port: they satisfy it structurally, and
mypy checks the match at the composition root.
"""
