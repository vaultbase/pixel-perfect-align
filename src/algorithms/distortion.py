"""
Lens distortion estimation and correction
"""

import numpy as np
import cv2
from scipy import optimize
from typing import Dict, Any, Tuple, Optional

from ..models.transform import Transform


class DistortionEstimator:
    """
    Estimates and corrects lens distortion parameters
    """
    
    def __init__(self):
        self.grid_size = (9, 6)  # Checkerboard size for calibration
        self.square_size = 30.0  # Size of checkerboard squares in pixels
    
    def estimate(
        self,
        image: np.ndarray,
        transform: Transform,
        known_distortion: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Estimate distortion parameters from image
        """
        if known_distortion is not None:
            return known_distortion
        
        # Try to detect straight lines for distortion estimation
        edges = self._detect_edges(image)
        lines = self._detect_lines(edges)
        
        if len(lines) > 10:
            # Estimate from straight line deviation
            params = self._estimate_from_lines(lines, image.shape)
        else:
            # Use default parameters for medium format
            params = self._default_medium_format_params()
        
        return params
    
    def undistort(
        self,
        image: np.ndarray,
        distortion_params: Dict[str, Any]
    ) -> np.ndarray:
        """
        Apply distortion correction to image
        """
        h, w = image.shape[:2]
        
        # Get camera matrix
        K = distortion_params.get('camera_matrix', self._default_camera_matrix(w, h))
        
        # Get distortion coefficients
        dist_coeffs = np.array([
            distortion_params.get('k1', 0.0),
            distortion_params.get('k2', 0.0),
            distortion_params.get('p1', 0.0),
            distortion_params.get('p2', 0.0),
            distortion_params.get('k3', 0.0)
        ])
        
        # Compute undistortion maps
        new_K = K.copy()
        map1, map2 = cv2.initUndistortRectifyMap(
            K, dist_coeffs, None, new_K, (w, h), cv2.CV_32FC1
        )
        
        # Apply undistortion
        undistorted = cv2.remap(
            image, map1, map2,
            interpolation=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        return undistorted
    
    def refine_distortion(
        self,
        images: list,
        initial_params: Dict[str, Any],
        transforms: list
    ) -> Dict[str, Any]:
        """
        Refine distortion parameters using multiple images
        """
        # Collect feature points from all images
        all_points = []
        all_lines = []
        
        for img in images:
            edges = self._detect_edges(img)
            lines = self._detect_lines(edges)
            all_lines.extend(lines)
        
        if len(all_lines) < 20:
            return initial_params
        
        # Optimize distortion parameters
        def objective(x):
            params = self._vector_to_params(x)
            
            total_error = 0.0
            for lines_subset in self._batch_lines(all_lines, 50):
                error = self._line_straightness_error(
                    lines_subset,
                    params,
                    images[0].shape
                )
                total_error += error
            
            return total_error
        
        # Initial parameters
        x0 = self._params_to_vector(initial_params)
        
        # Bounds for parameters
        bounds = [
            (-0.5, 0.5),  # k1
            (-0.5, 0.5),  # k2
            (-0.1, 0.1),  # p1
            (-0.1, 0.1),  # p2
            (-0.5, 0.5),  # k3
        ]
        
        # Optimize
        result = optimize.minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 50}
        )
        
        # Convert back to parameters
        refined_params = self._vector_to_params(result.x)
        refined_params['camera_matrix'] = initial_params['camera_matrix']
        
        return refined_params
    
    def _detect_edges(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        """
        Detect edges in image
        """
        if image.ndim == 3:
            gray = cv2.cvtColor(
                (image * 255).astype(np.uint8),
                cv2.COLOR_RGB2GRAY
            )
        else:
            gray = (image * 255).astype(np.uint8)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        
        return edges
    
    def _detect_lines(
        self,
        edges: np.ndarray
    ) -> list:
        """
        Detect lines using Hough transform
        """
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=100,
            maxLineGap=10
        )
        
        if lines is None:
            return []
        
        # Convert to list of line segments
        line_segments = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            line_segments.append({
                'p1': np.array([x1, y1]),
                'p2': np.array([x2, y2]),
                'length': np.sqrt((x2-x1)**2 + (y2-y1)**2)
            })
        
        # Filter out short lines
        line_segments = [l for l in line_segments if l['length'] > 50]
        
        return line_segments
    
    def _estimate_from_lines(
        self,
        lines: list,
        image_shape: Tuple[int, int]
    ) -> Dict[str, Any]:
        """
        Estimate distortion from line straightness
        """
        h, w = image_shape[:2]
        
        # Initial camera matrix
        K = self._default_camera_matrix(w, h)
        
        # Group lines by orientation
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            angle = np.arctan2(
                line['p2'][1] - line['p1'][1],
                line['p2'][0] - line['p1'][0]
            )
            
            if abs(angle) < np.pi/6 or abs(angle - np.pi) < np.pi/6:
                horizontal_lines.append(line)
            elif abs(angle - np.pi/2) < np.pi/6 or abs(angle + np.pi/2) < np.pi/6:
                vertical_lines.append(line)
        
        # Estimate distortion by minimizing line curvature
        def objective(x):
            k1, k2, p1, p2, k3 = x
            
            dist_coeffs = np.array([k1, k2, p1, p2, k3])
            
            # Compute error for horizontal and vertical lines
            error = 0.0
            
            for line in horizontal_lines + vertical_lines:
                # Sample points along line
                t = np.linspace(0, 1, 10)
                points = line['p1'] + t[:, np.newaxis] * (line['p2'] - line['p1'])
                
                # Undistort points
                undistorted = cv2.undistortPoints(
                    points.reshape(-1, 1, 2),
                    K,
                    dist_coeffs
                )
                undistorted = undistorted.reshape(-1, 2)
                
                # Measure deviation from straight line
                if len(undistorted) > 2:
                    # Fit line to undistorted points
                    vx, vy, x0, y0 = cv2.fitLine(
                        undistorted,
                        cv2.DIST_L2,
                        0, 0.01, 0.01
                    )
                    
                    # Compute distances to line
                    distances = []
                    for pt in undistorted:
                        dist = abs((pt[0] - x0) * vy - (pt[1] - y0) * vx)
                        distances.append(dist)
                    
                    error += np.mean(distances)
            
            return error
        
        # Initial guess
        x0 = [0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Bounds
        bounds = [
            (-0.3, 0.3),  # k1
            (-0.3, 0.3),  # k2
            (-0.05, 0.05),  # p1
            (-0.05, 0.05),  # p2
            (-0.3, 0.3),  # k3
        ]
        
        # Optimize
        result = optimize.minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 30}
        )
        
        k1, k2, p1, p2, k3 = result.x
        
        return {
            'camera_matrix': K,
            'k1': k1,
            'k2': k2,
            'p1': p1,
            'p2': p2,
            'k3': k3
        }
    
    def _default_camera_matrix(
        self,
        width: int,
        height: int
    ) -> np.ndarray:
        """
        Default camera matrix for medium format
        """
        # Assume 45mm lens on GFX100S II (35mm equivalent ~35mm)
        focal_length = max(width, height) * 1.2
        
        K = np.array([
            [focal_length, 0, width / 2],
            [0, focal_length, height / 2],
            [0, 0, 1]
        ], dtype=np.float32)
        
        return K
    
    def _default_medium_format_params(self) -> Dict[str, Any]:
        """
        Default distortion parameters for medium format
        """
        return {
            'k1': -0.05,  # Slight barrel distortion
            'k2': 0.01,
            'p1': 0.0,
            'p2': 0.0,
            'k3': 0.0
        }
    
    def _params_to_vector(self, params: Dict[str, Any]) -> np.ndarray:
        """Convert parameters to vector"""
        return np.array([
            params.get('k1', 0.0),
            params.get('k2', 0.0),
            params.get('p1', 0.0),
            params.get('p2', 0.0),
            params.get('k3', 0.0)
        ])
    
    def _vector_to_params(self, x: np.ndarray) -> Dict[str, Any]:
        """Convert vector to parameters"""
        return {
            'k1': x[0],
            'k2': x[1],
            'p1': x[2],
            'p2': x[3],
            'k3': x[4]
        }
    
    def _line_straightness_error(
        self,
        lines: list,
        params: Dict[str, Any],
        image_shape: Tuple[int, int]
    ) -> float:
        """
        Compute error based on line straightness after undistortion
        """
        h, w = image_shape[:2]
        K = params.get('camera_matrix', self._default_camera_matrix(w, h))
        
        dist_coeffs = np.array([
            params.get('k1', 0.0),
            params.get('k2', 0.0),
            params.get('p1', 0.0),
            params.get('p2', 0.0),
            params.get('k3', 0.0)
        ])
        
        total_error = 0.0
        
        for line in lines:
            # Sample points along line
            n_samples = max(10, int(line['length'] / 20))
            t = np.linspace(0, 1, n_samples)
            points = line['p1'] + t[:, np.newaxis] * (line['p2'] - line['p1'])
            
            # Undistort points
            undistorted = cv2.undistortPoints(
                points.reshape(-1, 1, 2).astype(np.float32),
                K,
                dist_coeffs,
                P=K
            )
            undistorted = undistorted.reshape(-1, 2)
            
            # Measure deviation from straight line
            if len(undistorted) > 2:
                # Fit line
                [vx], [vy], [x0], [y0] = cv2.fitLine(
                    undistorted.astype(np.float32),
                    cv2.DIST_L2,
                    0, 0.01, 0.01
                )
                
                # Compute perpendicular distances
                for pt in undistorted:
                    dist = abs((pt[0] - x0) * vy - (pt[1] - y0) * vx)
                    total_error += dist
        
        return total_error / max(len(lines), 1)
    
    def _batch_lines(self, lines: list, batch_size: int) -> list:
        """Batch lines for processing"""
        batches = []
        for i in range(0, len(lines), batch_size):
            batches.append(lines[i:i+batch_size])
        return batches