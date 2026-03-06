"""Custom exception hierarchy for Killswitch Engage.

Using specific exception types instead of bare ``Exception`` lets callers
handle predictable failure modes cleanly and prevents swallowing unexpected
errors (see SECURITY_CHECKLIST § 8 – Revisão manual de autenticação/validação).

Example:
    >>> from src.utils.exceptions import ModelNotLoadedError
    >>> raise ModelNotLoadedError("classifier")
"""

from __future__ import annotations

# ── Base ─────────────────────────────────────────────────────────────────────


class KillswitchError(Exception):
    """Base exception for all Killswitch Engage errors."""


# ── Model layer ───────────────────────────────────────────────────────────────


class ModelNotLoadedError(KillswitchError):
    """Raised when a model artefact is required but has not been loaded.

    Args:
        model_name (str): Descriptive name of the missing model.

    Example:
        >>> raise ModelNotLoadedError("LightFM ranker")
        ModelNotLoadedError: LightFM ranker model is not loaded.
        Check that the artefact file exists and the service started correctly.
    """

    def __init__(self, model_name: str) -> None:
        super().__init__(
            f"{model_name} model is not loaded. "
            "Check that the artefact file exists and the service started correctly."
        )
        self.model_name = model_name


class ModelArtefactNotFoundError(KillswitchError):
    """Raised when an expected model file is missing from disk.

    Args:
        path (str): Filesystem path that was expected to exist.
    """

    def __init__(self, path: str) -> None:
        super().__init__(f"Model artefact not found at: {path}")
        self.path = path


# ── Recommendation layer ──────────────────────────────────────────────────────


class InvalidModeError(KillswitchError):
    """Raised when an unknown recommendation mode is requested.

    Args:
        mode (str): The invalid mode string supplied by the caller.
        valid_modes (tuple[str, ...]): Modes that are accepted.
    """

    VALID_MODES = ("conservador", "equilibrado", "aventureiro")

    def __init__(self, mode: str) -> None:
        super().__init__(
            f"'{mode}' is not a valid recommendation mode. "
            f"Use one of: {sorted(self.VALID_MODES)}"
        )
        self.mode = mode


class NoRecommendationsError(KillswitchError):
    """Raised when the recommender produces an empty result set.

    Args:
        user_id (str): The user identifier for which recommendations were sought.
    """

    def __init__(self, user_id: str) -> None:
        super().__init__(f"No recommendations could be generated for user '{user_id}'.")
        self.user_id = user_id


# ── User / data layer ─────────────────────────────────────────────────────────


class UserNotFoundError(KillswitchError):
    """Raised when a user ID does not exist in the database.

    Args:
        user_id (str): The user identifier that was not found.
    """

    def __init__(self, user_id: str) -> None:
        # Security: do not expose internal system details in error message
        super().__init__(f"User '{user_id}' was not found.")
        self.user_id = user_id


class DataValidationError(KillswitchError):
    """Raised when input data fails a statistical or schema validation check.

    Args:
        detail (str): Human-readable description of the validation failure.
    """

    def __init__(self, detail: str) -> None:
        super().__init__(f"Data validation failed: {detail}")
        self.detail = detail


# ── Security helpers ──────────────────────────────────────────────────────────


class RateLimitExceededError(KillswitchError):
    """Raised (or caught) when a client exceeds the API rate limit."""


class AuthenticationError(KillswitchError):
    """Raised when authentication credentials are missing or invalid.

    Never include the actual credential value in this exception message.
    """

    def __init__(self) -> None:
        super().__init__(
            "Authentication failed. Check your credentials. "
            "If this is unexpected, contact the administrator."
        )
