"""
Image and metadata models
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path
import numpy as np


@dataclass
class ImageMetadata:
    """Metadata for an image"""
    filename: str
    path: Path
    width: int
    height: int
    channels: int
    bit_depth: int
    color_space: str
    exif_data: Optional[Dict[str, Any]] = None
    
    @property
    def is_raw(self) -> bool:
        """Check if image is raw format"""
        return self.path.suffix.lower() in {'.raw', '.dng', '.raf', '.nef', '.cr2', '.arw'}
    
    @property
    def megapixels(self) -> float:
        """Get megapixel count"""
        return (self.width * self.height) / 1_000_000


@dataclass
class AlignedImage:
    """Represents an aligned image with transform information"""
    image: np.ndarray
    mask: np.ndarray
    transform: 'Transform'
    distortion_params: Dict[str, Any]
    metadata: ImageMetadata
    canvas_position: tuple[int, int]
    
    @property
    def has_data(self) -> bool:
        """Check if image has valid data"""
        return np.any(self.mask > 0)
    
    def get_overlap_region(self, other: 'AlignedImage') -> np.ndarray:
        """Get mask of overlapping region with another image"""
        return self.mask & other.mask