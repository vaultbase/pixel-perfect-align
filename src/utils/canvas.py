"""
Canvas management utilities
"""

import numpy as np
from typing import List, Tuple

from ..models.transform import Transform


class CanvasManager:
    """
    Manages canvas size computation for aligned images
    """
    
    def compute_canvas_size(
        self,
        images: List[np.ndarray],
        transforms: List[Transform]
    ) -> Tuple[int, int]:
        """
        Compute canvas size to fit all transformed images
        """
        min_x = float('inf')
        min_y = float('inf')
        max_x = float('-inf')
        max_y = float('-inf')
        
        for img, transform in zip(images, transforms):
            h, w = img.shape[:2]
            
            # Transform corners
            corners = np.array([
                [0, 0],
                [w, 0],
                [w, h],
                [0, h]
            ])
            
            transformed_corners = transform.transform_points(corners)
            
            # Update bounds
            min_x = min(min_x, np.min(transformed_corners[:, 0]))
            min_y = min(min_y, np.min(transformed_corners[:, 1]))
            max_x = max(max_x, np.max(transformed_corners[:, 0]))
            max_y = max(max_y, np.max(transformed_corners[:, 1]))
        
        # Add margin
        margin = 50
        width = int(np.ceil(max_x - min_x)) + 2 * margin
        height = int(np.ceil(max_y - min_y)) + 2 * margin
        
        # Adjust transforms to account for offset
        for transform in transforms:
            transform.set_translation(
                transform.tx - min_x + margin,
                transform.ty - min_y + margin
            )
        
        return (height, width)
    
    def compute_overlap_regions(
        self,
        images: List[np.ndarray],
        transforms: List[Transform],
        canvas_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Compute overlap map showing how many images cover each pixel
        """
        h, w = canvas_size
        overlap_map = np.zeros((h, w), dtype=np.int32)
        
        for img, transform in zip(images, transforms):
            # Create mask
            mask = np.ones(img.shape[:2], dtype=np.uint8) * 255
            
            # Warp mask
            warped_mask, _ = transform.apply_to_image(mask, canvas_size)
            
            # Add to overlap map
            overlap_map[warped_mask > 0] += 1
        
        return overlap_map