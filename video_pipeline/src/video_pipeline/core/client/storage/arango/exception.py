"""ArangoDB client exceptions."""


class ArangoClientError(Exception):
    """Base exception for ArangoDB client errors."""
    pass


class ArangoConnectionError(ArangoClientError):
    """Raised when connection to ArangoDB fails."""
    pass


class ArangoQueryError(ArangoClientError):
    """Raised when an AQL query fails."""
    pass


class ArangoIndexError(ArangoClientError):
    """Raised when index creation or management fails."""
    pass