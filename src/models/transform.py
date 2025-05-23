"""
Transform and homography models
"""

import numpy as np
import cv2
from typing import Tuple, Optional


class Transform:
    """
    Represents a 2D transformation with translation, rotation, scale, and shear
    """
    
    def __init__(self):
        self.tx = 0.0
        self.ty = 0.0
        self.rotation = 0.0  # degrees
        self.scale = 1.0
        self.shear_x = 0.0
        self.shear_y = 0.0
        self._matrix = None
        self._inverse = None
    
    def set_translation(self, tx: float, ty: float):
        """Set translation parameters"""
        self.tx = tx
        self.ty = ty
        self._invalidate_cache()
    
    def set_rotation(self, angle: float):
        """Set rotation in degrees"""
        self.rotation = angle
        self._invalidate_cache()
    
    def set_scale(self, scale: float):
        """Set uniform scale"""
        self.scale = scale
        self._invalidate_cache()
    
    def set_shear(self, shear_x: float, shear_y: float):
        """Set shear parameters"""
        self.shear_x = shear_x
        self.shear_y = shear_y
        self._invalidate_cache()
    
    def get_matrix(self) -> np.ndarray:
        """Get 3x3 homography matrix"""
        if self._matrix is None:
            self._compute_matrix()
        return self._matrix
    
    def get_inverse(self) -> np.ndarray:
        """Get inverse transformation matrix"""
        if self._inverse is None:
            self._inverse = np.linalg.inv(self.get_matrix())
        return self._inverse
    
    def transform_point(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """Transform a single point"""
        p = np.array([point[0], point[1], 1.0])
        transformed = self.get_matrix() @ p
        return (transformed[0] / transformed[2], transformed[1] / transformed[2])
    
    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Transform multiple points"""
        # Add homogeneous coordinate
        n_points = len(points)
        homogeneous = np.ones((n_points, 3))
        homogeneous[:, :2] = points
        
        # Transform
        transformed = homogeneous @ self.get_matrix().T
        
        # Normalize
        transformed[:, :2] /= transformed[:, 2:3]
        
        return transformed[:, :2]
    
    def compose(self, other: 'Transform') -> 'Transform':
        """Compose this transform with another"""
        result = Transform()
        result._matrix = self.get_matrix() @ other.get_matrix()
        result._decompose_matrix()
        return result
    
    def scale_by(self, factor: float) -> 'Transform':
        """Scale the transform parameters"""
        result = Transform()
        result.tx = self.tx * factor
        result.ty = self.ty * factor
        result.rotation = self.rotation
        result.scale = self.scale
        result.shear_x = self.shear_x
        result.shear_y = self.shear_y
        return result
    
    def apply_to_image(
        self,
        image: np.ndarray,
        output_size: Tuple[int, int],
        border_mode: int = cv2.BORDER_CONSTANT,
        border_value: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply transform to image
        Returns: (warped_image, mask)
        """
        h, w = image.shape[:2]
        out_h, out_w = output_size
        
        # Get transformation matrix
        M = self.get_matrix()[:2, :]  # 2x3 matrix for cv2.warpAffine
        
        # Warp image
        flags = cv2.INTER_CUBIC
        warped = cv2.warpAffine(
            image, M, (out_w, out_h),
            flags=flags,
            borderMode=border_mode,
            borderValue=border_value
        )
        
        # Create mask
        mask = np.ones((h, w), dtype=np.uint8) * 255
        warped_mask = cv2.warpAffine(
            mask, M, (out_w, out_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        return warped, warped_mask
    
    def _compute_matrix(self):
        """Compute transformation matrix from parameters"""
        # Translation matrix
        T = np.array([
            [1, 0, self.tx],
            [0, 1, self.ty],
            [0, 0, 1]
        ])
        
        # Rotation matrix
        angle_rad = np.radians(self.rotation)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        R = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])
        
        # Scale matrix
        S = np.array([
            [self.scale, 0, 0],
            [0, self.scale, 0],
            [0, 0, 1]
        ])
        
        # Shear matrix
        H = np.array([
            [1, self.shear_x, 0],
            [self.shear_y, 1, 0],
            [0, 0, 1]
        ])
        
        # Compose: T * R * S * H
        self._matrix = T @ R @ S @ H
    
    def _decompose_matrix(self):
        """Decompose matrix into parameters (approximate)"""
        M = self._matrix
        
        # Extract translation
        self.tx = M[0, 2]
        self.ty = M[1, 2]
        
        # Extract rotation and scale
        a = M[0, 0]
        b = M[0, 1]
        c = M[1, 0]
        d = M[1, 1]
        
        # Scale
        sx = np.sqrt(a*a + c*c)
        sy = np.sqrt(b*b + d*d)
        self.scale = (sx + sy) / 2  # Average scale
        
        # Rotation
        self.rotation = np.degrees(np.arctan2(c, a))
        
        # Shear (simplified)
        self.shear_x = 0.0
        self.shear_y = 0.0
    
    def _invalidate_cache(self):
        """Invalidate cached matrices"""
        self._matrix = None
        self._inverse = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'tx': float(self.tx),
            'ty': float(self.ty),
            'rotation': float(self.rotation),
            'scale': float(self.scale),
            'shear_x': float(self.shear_x),
            'shear_y': float(self.shear_y),
            'matrix': self.get_matrix().tolist()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Transform':
        """Create from dictionary"""
        t = cls()
        t.tx = data['tx']
        t.ty = data['ty']
        t.rotation = data['rotation']
        t.scale = data['scale']
        t.shear_x = data.get('shear_x', 0.0)
        t.shear_y = data.get('shear_y', 0.0)
        return t


class Homography:
    """
    Full projective homography transformation
    """
    
    def __init__(self, matrix: Optional[np.ndarray] = None):
        if matrix is None:
            self.matrix = np.eye(3, dtype=np.float64)
        else:
            self.matrix = matrix.copy()
    
    def transform_point(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """Transform a single point"""
        p = np.array([point[0], point[1], 1.0])
        transformed = self.matrix @ p
        return (transformed[0] / transformed[2], transformed[1] / transformed[2])
    
    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Transform multiple points"""
        # Add homogeneous coordinate
        n_points = len(points)
        homogeneous = np.ones((n_points, 3))
        homogeneous[:, :2] = points
        
        # Transform
        transformed = homogeneous @ self.matrix.T
        
        # Normalize
        transformed[:, :2] /= transformed[:, 2:3]
        
        return transformed[:, :2]
    
    def apply_to_image(
        self,
        image: np.ndarray,
        output_size: Tuple[int, int],
        border_mode: int = cv2.BORDER_CONSTANT,
        border_value: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply homography to image
        Returns: (warped_image, mask)
        """
        h, w = image.shape[:2]
        out_h, out_w = output_size
        
        # Warp image
        flags = cv2.INTER_CUBIC
        warped = cv2.warpPerspective(
            image, self.matrix, (out_w, out_h),
            flags=flags,
            borderMode=border_mode,
            borderValue=border_value
        )
        
        # Create mask
        mask = np.ones((h, w), dtype=np.uint8) * 255
        warped_mask = cv2.warpPerspective(
            mask, self.matrix, (out_w, out_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        return warped, warped_mask
    
    def inverse(self) -> 'Homography':
        """Get inverse homography"""
        return Homography(np.linalg.inv(self.matrix))
    
    def compose(self, other: 'Homography') -> 'Homography':
        """Compose with another homography"""
        return Homography(self.matrix @ other.matrix)
    
    @classmethod
    def from_points(
        cls,
        src_points: np.ndarray,
        dst_points: np.ndarray
    ) -> 'Homography':
        """Estimate homography from point correspondences"""
        matrix, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC)
        return cls(matrix)
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'matrix': self.matrix.tolist()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Homography':
        """Create from dictionary"""
        return cls(np.array(data['matrix']))