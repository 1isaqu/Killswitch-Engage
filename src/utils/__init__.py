"""Utils package for Killswitch Engage."""

from src.utils.exceptions import (
    AuthenticationError,
    DataValidationError,
    InvalidModeError,
    KillswitchError,
    ModelArtefactNotFoundError,
    ModelNotLoadedError,
    NoRecommendationsError,
    UserNotFoundError,
)
from src.utils.logger import get_logger

__all__ = [
    "get_logger",
    "KillswitchError",
    "ModelNotLoadedError",
    "ModelArtefactNotFoundError",
    "InvalidModeError",
    "NoRecommendationsError",
    "UserNotFoundError",
    "DataValidationError",
    "AuthenticationError",
]
