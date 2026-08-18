"""DC-Ops RAG: grounded question answering over datacenter facilities documentation.

The package is layered (see docs/architecture.md and docs/adr/0001-clean-architecture.md):

* ``dcrag.domain`` — entities and ports. Pure Python, no third-party imports.
* ``dcrag.application`` — use cases orchestrating ports. Knows no concrete technology.
* ``dcrag.infrastructure`` — adapters implementing the ports (Qdrant, Ollama, FAISS, fakes).
* ``dcrag.api`` / ``dcrag.agent`` / ``dcrag.eval`` — delivery mechanisms and tooling.

Dependencies point inwards only: infrastructure imports domain, never the other way round.
"""

__version__ = "0.1.0"
