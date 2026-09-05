"""
Satellite Provider Service
Orchestrates satellite imagery providers, caching, and coordinate normalization.
"""

import os
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from PIL import Image

from .provider_base import SatelliteProvider, AOIBoundingBox, SatelliteSceneMetadata
from .sentinel_provider import SentinelSTACProvider
from .nasa_gibs_provider import NASAGIBSProvider

CACHE_DIR = Path(__file__).parent.parent / "cache" / "satellite"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class SatelliteProviderService:
    """Manages satellite imagery providers with caching and fallback."""

    def __init__(self):
        self._providers: Dict[str, SatelliteProvider] = {
            "sentinel-stac": SentinelSTACProvider(),
            "nasa-gibs": NASAGIBSProvider(),
        }
        # Configured default provider
        self.default_provider_key = os.getenv("SATELLITE_PROVIDER", "sentinel-stac").lower()
        if self.default_provider_key not in self._providers:
            self.default_provider_key = "sentinel-stac"

    def get_provider(self, key: Optional[str] = None) -> SatelliteProvider:
        key = (key or self.default_provider_key).lower()
        return self._providers.get(key, self._providers["sentinel-stac"])

    def list_providers(self) -> List[Dict[str, str]]:
        return [
            {"id": k, "name": p.provider_name}
            for k, p in self._providers.items()
        ]

    def _cache_key(self, aoi: AOIBoundingBox, tag: str, date_str: str) -> str:
        s = f"{aoi.north:.4f}_{aoi.south:.4f}_{aoi.east:.4f}_{aoi.west:.4f}_{tag}_{date_str}"
        return hashlib.md5(s.encode("utf-8")).hexdigest()

    def get_latest(
        self,
        aoi: AOIBoundingBox,
        max_cloud_cover: float = 20.0,
        search_days: int = 30,
        provider_key: Optional[str] = None,
    ) -> Tuple[Image.Image, SatelliteSceneMetadata]:
        """Fetch latest usable satellite image."""
        provider = self.get_provider(provider_key)
        return provider.get_latest_image(
            aoi=aoi,
            max_cloud_cover=max_cloud_cover,
            search_days=search_days,
        )

    def get_historical(
        self,
        aoi: AOIBoundingBox,
        target_date: str,
        date_range_days: int = 14,
        max_cloud_cover: float = 25.0,
        provider_key: Optional[str] = None,
    ) -> Tuple[Image.Image, SatelliteSceneMetadata]:
        """Fetch historical satellite image."""
        provider = self.get_provider(provider_key)
        return provider.get_historical_image(
            aoi=aoi,
            target_date=target_date,
            date_range_days=date_range_days,
            max_cloud_cover=max_cloud_cover,
        )


# Singleton instance
satellite_service = SatelliteProviderService()

