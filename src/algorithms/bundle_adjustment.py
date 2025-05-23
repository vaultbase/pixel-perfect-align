"""
Global bundle adjustment for simultaneous optimization of all image parameters
"""

import numpy as np
from scipy import optimize, sparse
from typing import List, Dict, Any, Tuple
import cv2

from ..models.transform import Transform


class BundleAdjuster:
    """
    Implements global bundle adjustment for multi-image alignment
    """
    
    def __init__(
        self,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        regularization: float = 0.01
    ):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.regularization = regularization
        self.feature_detector = cv2.SIFT_create(nfeatures=5000)
    
    def extract_features(
        self,
        image: np.ndarray
    ) -> Dict[str, Any]:
        """
        Extract SIFT features for bundle adjustment
        """
        if image.ndim == 3:
            gray = cv2.cvtColor(
                (image * 255).astype(np.uint8),
                cv2.COLOR_RGB2GRAY
            )
        else:
            gray = (image * 255).astype(np.uint8)
        
        # Detect keypoints and compute descriptors
        keypoints, descriptors = self.feature_detector.detectAndCompute(gray, None)
        
        # Convert keypoints to numpy array
        points = np.array([kp.pt for kp in keypoints])
        scales = np.array([kp.size for kp in keypoints])
        angles = np.array([kp.angle for kp in keypoints])
        
        return {
            'points': points,
            'descriptors': descriptors,
            'scales': scales,
            'angles': angles
        }
    
    def optimize(
        self,
        all_features: List[Dict[str, Any]],
        initial_transforms: List[Transform],
        distortion_params: List[Dict[str, Any]],
        overlap_ratio: float
    ) -> List[Transform]:
        """
        Perform global bundle adjustment
        """
        n_images = len(all_features)
        
        # Find feature matches between overlapping images
        matches = self._find_all_matches(all_features, overlap_ratio)
        
        # Build correspondence graph
        correspondence_graph = self._build_correspondence_graph(
            matches, n_images
        )
        
        # Initialize optimization parameters
        params = self._transforms_to_params(initial_transforms)
        
        # Define objective function
        def objective(x):
            transforms = self._params_to_transforms(x, n_images)
            error = self._compute_reprojection_error(
                transforms,
                all_features,
                correspondence_graph
            )
            
            # Add regularization
            reg_term = self.regularization * np.sum(x ** 2)
            
            return error + reg_term
        
        # Define Jacobian
        def jacobian(x):
            return self._compute_jacobian(
                x,
                all_features,
                correspondence_graph,
                n_images
            )
        
        # Optimize
        result = optimize.minimize(
            objective,
            params,
            method='L-BFGS-B',
            jac=jacobian,
            options={
                'maxiter': self.max_iterations,
                'ftol': self.tolerance
            }
        )
        
        # Convert back to transforms
        optimized_transforms = self._params_to_transforms(
            result.x, n_images
        )
        
        # Apply additional constraints
        optimized_transforms = self._apply_constraints(
            optimized_transforms,
            correspondence_graph
        )
        
        return optimized_transforms
    
    def _find_all_matches(
        self,
        all_features: List[Dict[str, Any]],
        overlap_ratio: float
    ) -> List[Dict[str, Any]]:
        """
        Find feature matches between all overlapping image pairs
        """
        n_images = len(all_features)
        matches = []
        
        # FLANN matcher for efficient matching
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Match all pairs
        for i in range(n_images):
            for j in range(i + 1, n_images):
                # Check if images might overlap (simple heuristic)
                if abs(i - j) > 3:  # Skip if too far apart
                    continue
                
                # Match descriptors
                if (all_features[i]['descriptors'] is not None and
                    all_features[j]['descriptors'] is not None):
                    
                    raw_matches = flann.knnMatch(
                        all_features[i]['descriptors'],
                        all_features[j]['descriptors'],
                        k=2
                    )
                    
                    # Lowe's ratio test
                    good_matches = []
                    for match_pair in raw_matches:
                        if len(match_pair) == 2:
                            m, n = match_pair
                            if m.distance < 0.7 * n.distance:
                                good_matches.append(m)
                    
                    if len(good_matches) > 10:
                        matches.append({
                            'img1': i,
                            'img2': j,
                            'matches': good_matches,
                            'points1': all_features[i]['points'],
                            'points2': all_features[j]['points']
                        })
        
        return matches
    
    def _build_correspondence_graph(
        self,
        matches: List[Dict[str, Any]],
        n_images: int
    ) -> Dict[str, Any]:
        """
        Build graph of feature correspondences
        """
        # Track points across multiple images
        tracks = []
        point_to_track = [{} for _ in range(n_images)]
        
        for match_set in matches:
            img1 = match_set['img1']
            img2 = match_set['img2']
            
            for m in match_set['matches']:
                pt1_idx = m.queryIdx
                pt2_idx = m.trainIdx
                
                # Check if points already belong to tracks
                track_id1 = point_to_track[img1].get(pt1_idx, None)
                track_id2 = point_to_track[img2].get(pt2_idx, None)
                
                if track_id1 is None and track_id2 is None:
                    # Create new track
                    track = {
                        'points': {
                            img1: pt1_idx,
                            img2: pt2_idx
                        }
                    }
                    tracks.append(track)
                    track_id = len(tracks) - 1
                    point_to_track[img1][pt1_idx] = track_id
                    point_to_track[img2][pt2_idx] = track_id
                
                elif track_id1 is not None and track_id2 is None:
                    # Add to existing track
                    tracks[track_id1]['points'][img2] = pt2_idx
                    point_to_track[img2][pt2_idx] = track_id1
                
                elif track_id1 is None and track_id2 is not None:
                    # Add to existing track
                    tracks[track_id2]['points'][img1] = pt1_idx
                    point_to_track[img1][pt1_idx] = track_id2
                
                elif track_id1 != track_id2:
                    # Merge tracks
                    track1 = tracks[track_id1]
                    track2 = tracks[track_id2]
                    
                    # Merge into track1
                    for img, pt_idx in track2['points'].items():
                        if img not in track1['points']:
                            track1['points'][img] = pt_idx
                            point_to_track[img][pt_idx] = track_id1
                    
                    # Mark track2 as merged
                    track2['merged'] = True
        
        # Filter out merged tracks
        valid_tracks = [t for t in tracks if not t.get('merged', False)]
        
        return {
            'tracks': valid_tracks,
            'n_images': n_images
        }
    
    def _transforms_to_params(
        self,
        transforms: List[Transform]
    ) -> np.ndarray:
        """
        Convert transforms to optimization parameters
        """
        params = []
        
        # First image is reference (no parameters)
        for i in range(1, len(transforms)):
            t = transforms[i]
            params.extend([
                t.tx, t.ty,           # Translation
                t.rotation,           # Rotation
                np.log(t.scale),      # Log scale for better optimization
                t.shear_x, t.shear_y  # Shear
            ])
        
        return np.array(params)
    
    def _params_to_transforms(
        self,
        params: np.ndarray,
        n_images: int
    ) -> List[Transform]:
        """
        Convert optimization parameters to transforms
        """
        transforms = [Transform()]  # Identity for first image
        
        params_per_image = 6
        
        for i in range(1, n_images):
            idx = (i - 1) * params_per_image
            
            t = Transform()
            t.set_translation(params[idx], params[idx + 1])
            t.set_rotation(params[idx + 2])
            t.set_scale(np.exp(params[idx + 3]))
            t.set_shear(params[idx + 4], params[idx + 5])
            
            transforms.append(t)
        
        return transforms
    
    def _compute_reprojection_error(
        self,
        transforms: List[Transform],
        all_features: List[Dict[str, Any]],
        correspondence_graph: Dict[str, Any]
    ) -> float:
        """
        Compute total reprojection error
        """
        total_error = 0.0
        n_points = 0
        
        for track in correspondence_graph['tracks']:
            if len(track['points']) < 2:
                continue
            
            # Get all points in track
            points = []
            for img_idx, pt_idx in track['points'].items():
                pt = all_features[img_idx]['points'][pt_idx]
                transformed_pt = transforms[img_idx].transform_point(pt)
                points.append(transformed_pt)
            
            points = np.array(points)
            
            # Compute centroid
            centroid = np.mean(points, axis=0)
            
            # Compute distances to centroid
            distances = np.linalg.norm(points - centroid, axis=1)
            
            # Robust error (Huber loss)
            threshold = 5.0
            errors = np.where(
                distances < threshold,
                distances ** 2,
                2 * threshold * distances - threshold ** 2
            )
            
            total_error += np.sum(errors)
            n_points += len(points)
        
        return total_error / max(n_points, 1)
    
    def _compute_jacobian(
        self,
        params: np.ndarray,
        all_features: List[Dict[str, Any]],
        correspondence_graph: Dict[str, Any],
        n_images: int
    ) -> np.ndarray:
        """
        Compute Jacobian of reprojection error
        """
        n_params = len(params)
        jacobian = np.zeros(n_params)
        
        # Finite differences
        epsilon = 1e-6
        
        transforms = self._params_to_transforms(params, n_images)
        base_error = self._compute_reprojection_error(
            transforms, all_features, correspondence_graph
        )
        
        for i in range(n_params):
            params_plus = params.copy()
            params_plus[i] += epsilon
            
            transforms_plus = self._params_to_transforms(params_plus, n_images)
            error_plus = self._compute_reprojection_error(
                transforms_plus, all_features, correspondence_graph
            )
            
            jacobian[i] = (error_plus - base_error) / epsilon
        
        return jacobian
    
    def _apply_constraints(
        self,
        transforms: List[Transform],
        correspondence_graph: Dict[str, Any]
    ) -> List[Transform]:
        """
        Apply additional constraints to ensure global consistency
        """
        # Ensure first image remains fixed
        transforms[0] = Transform()
        
        # Apply loop closure constraints if detected
        # (simplified version - full implementation would detect loops)
        
        # Normalize scales to prevent drift
        scales = [t.scale for t in transforms]
        mean_scale = np.mean(scales)
        
        for t in transforms:
            t.scale = t.scale / mean_scale
        
        return transforms