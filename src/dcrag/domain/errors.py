"""Domain errors.

Every failure the system can produce on purpose has a named type. Adapters translate
library-specific exceptions into these, so the application layer never catches a
``qdrant_client`` or ``httpx`` exception and the API can map errors to status codes
without knowing which technology failed.
"""


class DcragError(Exception):
    """Base class for every error this system raises deliberately."""


class DomainValidationError(DcragError):
    """An entity was constructed in a state the domain forbids.

    Raised from ``__post_init__`` validators. Reaching this means a programming error
    upstream, not a user error, so it is not something the API should surface verbatim.
    """


class ConfigurationError(DcragError):
    """The system was wired or configured in a way that cannot work.

    Example: a retrieval mode requiring a lexical index while no lexical index was
    injected into the service.
    """


class UngroundedCitationError(DcragError):
    """A generated answer cited a context block that was never retrieved.

    This is the load-bearing safety check of the whole system: if the model invents a
    citation, the answer is not grounded and must not be served as if it were.
    """


class RetrievalError(DcragError):
    """A retrieval backend failed to answer a query."""


class GenerationError(DcragError):
    """A generator backend failed to produce a completion."""
