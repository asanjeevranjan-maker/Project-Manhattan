"""
Satellite Provider Base Module
Provides abstract base classes and data structures for satellite imagery providers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List
from PIL import Image


@dataclass
class AOIBoundingBox:
    """
    Geographic Area of Interest (AOI) bounding box.
    Coordinates in standard WGS84 (EPSG:4326) degrees.
    """
    north: float
    south: float
    east: float
    west: float

    def validate(self, max_span_degrees: float = 0.25) -> None:
        """Ensure coordinate sanity and prevent fetching gigantic regions."""
        if not (-90.0 <= self.south < self.north <= 90.0):
            raise ValueError(
                f"Invalid latitude range: south={self.south}, north={self.north}. "
                "Must be between -90 and 90 with north > south."
            )
        if not (-180.0 <= self.west <= 180.0 and -180.0 <= self.east <= 180.0):
            raise ValueError(
                f"Invalid longitude range: west={self.west}, east={self.east}. "
                "Must be between -180 and 180."
            )
        if self.east <= self.west and not (self.west > 0 and self.east < 0):
            raise ValueError("east must be greater than west")

        lat_span = abs(self.north - self.south)
        lon_span = abs(self.east - self.west)
        if lat_span > max_span_degrees or lon_span > max_span_degrees:
            raise ValueError(
                f"Requested AOI span ({lat_span:.3f}° lat, {lon_span:.3f}° lon) "
                f"exceeds maximum allowed size of {max_span_degrees}° (~{int(max_span_degrees * 111)} km). "
                "Please select a more focused area of interest."
            )

    @property
    def center(self) -> tuple[float, float]:
        """Returns (latitude, longitude) center of AOI."""
        return ((self.north + self.south) / 2.0, (self.east + self.west) / 2.0)

    def to_bbox_list(self) -> List[float]:
        """Returns [west, south, east, north] (GeoJSON / STAC standard)."""
        return [self.west, self.south, self.east, self.north]

    def to_dict(self) -> Dict[str, float]:
        return {
            "north": self.north,
            "south": self.south,
            "east": self.east,
            "west": self.west,
        }


@dataclass
class SatelliteSceneMetadata:
    """Standardized metadata for a satellite acquisition scene."""
    scene_id: str
    provider: str
    satellite: str
    acquisition_date: str  # YYYY-MM-DD or ISO8601
    acquisition_time: Optional[str] = None
    cloud_coverage: Optional[float] = None  # 0.0 to 100.0 percent
    resolution: str = "10m"
    data_freshness_days: Optional[int] = None
    preview_url: Optional[str] = None
    extra_properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sceneId": self.scene_id,
            "provider": self.provider,
            "satellite": self.satellite,
            "acquisitionDate": self.acquisition_date,
            "acquisitionTime": self.acquisition_time,
            "cloudCoverage": round(self.cloud_coverage, 1) if self.cloud_coverage is not None else None,
            "resolution": self.resolution,
            "dataFreshnessDays": self.data_freshness_days,
            "previewUrl": self.preview_url,
            "extraProperties": self.extra_properties,
        }


class SatelliteProvider(ABC):
    """Abstract Interface for any Satellite Imagery Provider."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider name."""
        pass

    @abstractmethod
    def search_images(
        self,
        aoi: AOIBoundingBox,
        start_date: str,
        end_date: str,
        max_cloud_cover: float = 25.0,
        limit: int = 10,
    ) -> List[SatelliteSceneMetadata]:
        """Search available scenes intersecting AOI within a date range."""
        pass

    @abstractmethod
    def get_latest_image(
        self,
        aoi: AOIBoundingBox,
        max_cloud_cover: float = 20.0,
        search_days: int = 30,
    ) -> tuple[Image.Image, SatelliteSceneMetadata]:
        """
        Search backwards from today's date to obtain the newest usable scene
        below the max_cloud_cover threshold, rendered for the AOI.
        """
        pass

    @abstractmethod
    def get_historical_image(
        self,
        aoi: AOIBoundingBox,
        target_date: str,
        date_range_days: int = 14,
        max_cloud_cover: float = 25.0,
    ) -> tuple[Image.Image, SatelliteSceneMetadata]:
        """
        Retrieve the best available scene close to target_date (+/- date_range_days)
        with lowest cloud coverage for the AOI.
        """
        pass

    @abstractmethod
    def download_or_render_image(
        self,
        scene: SatelliteSceneMetadata,
        aoi: AOIBoundingBox,
        target_size: tuple[int, int] = (640, 640),
    ) -> Image.Image:
        """Download or render RGB imagery for the scene cropped to the exact AOI."""
        pass

