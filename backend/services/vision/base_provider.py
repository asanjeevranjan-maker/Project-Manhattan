"""
Vision Provider Abstract Base Class & Domain Exceptions
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from .response_parser import SatelliteAnalysisStructured


class VisionProviderError(Exception):
    """Base exception for vision provider API errors."""
    def __init__(self, message: str, status_code: int = 500, provider: str = "unknown"):
        super().__init__(message)
        self.status_code = status_code
        self.provider = provider


class VisionProviderAuthError(VisionProviderError):
    """Raised when API key is missing or rejected."""
    pass


class VisionProviderRateLimitError(VisionProviderError):
    """Raised when upstream rate limit (HTTP 429) is encountered."""
    pass


class VisionProvider(ABC):
    """
    Abstract interface for multimodal vision analysis providers (Gemini, GLM).
    Enforces identical input and output contracts across all providers.
    """

    provider_name: str = "base"

    @abstractmethod
    async def analyze(
        self,
        image_bytes: bytes,
        mime_type: str,
        user_query: str,
        analysis_mode: str = "general",
        detection_context: Optional[Dict[str, Any]] = None,
        change_context: Optional[Dict[str, Any]] = None,
        image_metadata: Optional[Dict[str, Any]] = None,
        second_image_bytes: Optional[bytes] = None,
        second_mime_type: Optional[str] = None,
        temperature: float = 0.2,
        spatial_tile_label: Optional[str] = None,
        segmentation_summary: Optional[Dict[str, Any]] = None,
        land_cover: Optional[Dict[str, Any]] = None,
    ) -> SatelliteAnalysisStructured:
        """
        Executes multimodal analysis and returns a normalized SatelliteAnalysisStructured object.
        """
        pass
