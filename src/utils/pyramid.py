"""
Image pyramid utilities for multi-resolution processing
"""

import numpy as np
import cv2
from typing import List, Optional


class ImagePyramid:
    """
    Multi-resolution image pyramid
    """
    
    def __init__(
        self,
        image: np.ndarray,
        levels: int = 5,
        scale_factor: float = 2.0
    ):
        self.original = image
        self.levels = levels
        self.scale_factor = scale_factor
        self.pyramid = self._build_pyramid()
    
    def _build_pyramid(self) -> List[np.ndarray]:
        """
        Build Gaussian pyramid
        """
        pyramid = [self.original]
        current = self.original
        
        for level in range(1, self.levels):
            # Apply Gaussian blur before downsampling
            sigma = self.scale_factor / 2.0
            blurred = cv2.GaussianBlur(
                current,
                (0, 0),
                sigmaX=sigma,
                sigmaY=sigma
            )
            
            # Downsample
            new_h = int(current.shape[0] / self.scale_factor)
            new_w = int(current.shape[1] / self.scale_factor)
            
            downsampled = cv2.resize(
                blurred,
                (new_w, new_h),
                interpolation=cv2.INTER_LINEAR
            )
            
            pyramid.append(downsampled)
            current = downsampled
        
        return pyramid
    
    def get_level(self, level: int) -> np.ndarray:
        """
        Get image at specific pyramid level
        """
        if level < 0 or level >= len(self.pyramid):
            raise ValueError(f"Invalid pyramid level: {level}")
        
        return self.pyramid[level]
    
    @property
    def num_levels(self) -> int:
        """
        Get number of pyramid levels
        """
        return len(self.pyramid)
    
    def get_scale_factor(self, level: int) -> float:
        """
        Get scale factor for a level relative to original
        """
        return self.scale_factor ** level
    
    def upsample_to_level(
        self,
        image: np.ndarray,
        from_level: int,
        to_level: int
    ) -> np.ndarray:
        """
        Upsample image from one level to another
        """
        if from_level <= to_level:
            return image
        
        scale = self.get_scale_factor(to_level) / self.get_scale_factor(from_level)
        
        new_h = int(image.shape[0] * scale)
        new_w = int(image.shape[1] * scale)
        
        return cv2.resize(
            image,
            (new_w, new_h),
            interpolation=cv2.INTER_CUBIC
        )