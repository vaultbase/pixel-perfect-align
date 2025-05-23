"""
Attention-based patch matching for robust feature correspondence
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

from ..models.transform import Transform


class AttentionMatcher:
    """
    Implements attention-based patch matching for alignment refinement
    """
    
    def __init__(
        self,
        patch_size: int = 64,
        stride: int = 32,
        num_scales: int = 3,
        attention_heads: int = 8
    ):
        self.patch_size = patch_size
        self.stride = stride
        self.num_scales = num_scales
        self.attention_heads = attention_heads
        self.pca = PCA(n_components=128)
    
    def extract_patches(
        self,
        image: np.ndarray,
        mask: np.ndarray = None
    ) -> Dict[str, Any]:
        """
        Extract multi-scale patches from image
        """
        if image.ndim == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image.copy()
        
        patches = []
        positions = []
        scales = []
        
        # Extract patches at multiple scales
        for scale in range(self.num_scales):
            scale_factor = 2 ** scale
            
            # Downsample image
            if scale > 0:
                scaled_img = self._downsample(gray, scale_factor)
            else:
                scaled_img = gray
            
            # Extract patches
            h, w = scaled_img.shape
            for y in range(0, h - self.patch_size + 1, self.stride):
                for x in range(0, w - self.patch_size + 1, self.stride):
                    patch = scaled_img[y:y+self.patch_size, x:x+self.patch_size]
                    
                    # Check if patch is valid (not all black)
                    if np.std(patch) > 1e-6:
                        patches.append(patch.flatten())
                        positions.append((x * scale_factor, y * scale_factor))
                        scales.append(scale)
        
        patches = np.array(patches)
        
        # Compute patch descriptors
        descriptors = self._compute_descriptors(patches)
        
        return {
            'patches': patches,
            'descriptors': descriptors,
            'positions': positions,
            'scales': scales
        }
    
    def match_and_refine(
        self,
        target_image: np.ndarray,
        reference_patches: List[Tuple[int, Dict[str, Any]]],
        initial_transform: Transform
    ) -> Transform:
        """
        Match patches and refine transform
        """
        # Extract patches from target
        target_data = self.extract_patches(target_image)
        
        # Match patches across images
        all_matches = []
        
        for ref_idx, ref_data in reference_patches:
            matches = self._match_patches(
                target_data,
                ref_data,
                initial_transform
            )
            all_matches.extend(matches)
        
        # Apply attention mechanism to weight matches
        weighted_matches = self._apply_attention(all_matches)
        
        # Estimate refined transform from matches
        refined_transform = self._estimate_transform(weighted_matches)
        
        return refined_transform
    
    def _compute_descriptors(
        self,
        patches: np.ndarray
    ) -> np.ndarray:
        """
        Compute robust patch descriptors
        """
        # Normalize patches
        normalized = []
        for patch in patches:
            p = patch - np.mean(patch)
            std = np.std(p)
            if std > 1e-6:
                p = p / std
            normalized.append(p)
        
        normalized = np.array(normalized)
        
        # Fit PCA if needed
        if not hasattr(self.pca, 'components_'):
            self.pca.fit(normalized)
        
        # Transform to lower dimension
        descriptors = self.pca.transform(normalized)
        
        # Add spatial frequency features
        freq_features = []
        for patch in patches:
            patch_2d = patch.reshape(self.patch_size, self.patch_size)
            fft = np.fft.fft2(patch_2d)
            freq_feat = np.array([
                np.mean(np.abs(fft)),
                np.std(np.abs(fft)),
                np.mean(np.angle(fft)),
                np.std(np.angle(fft))
            ])
            freq_features.append(freq_feat)
        
        freq_features = np.array(freq_features)
        
        # Concatenate all features
        descriptors = np.hstack([descriptors, freq_features])
        
        return descriptors
    
    def _match_patches(
        self,
        target_data: Dict[str, Any],
        reference_data: Dict[str, Any],
        initial_transform: Transform
    ) -> List[Dict[str, Any]]:
        """
        Find corresponding patches between images
        """
        matches = []
        
        # Compute descriptor distances
        distances = cdist(
            target_data['descriptors'],
            reference_data['descriptors'],
            metric='euclidean'
        )
        
        # For each target patch, find best matches
        for i in range(len(target_data['descriptors'])):
            # Get top-k matches
            k = min(5, len(reference_data['descriptors']))
            best_matches = np.argpartition(distances[i], k)[:k]
            
            for j in best_matches:
                # Check spatial consistency
                target_pos = target_data['positions'][i]
                ref_pos = reference_data['positions'][j]
                
                # Transform reference position using initial transform
                transformed_ref = initial_transform.transform_point(ref_pos)
                
                # Check if positions are reasonably close
                dist = np.linalg.norm(
                    np.array(target_pos) - np.array(transformed_ref)
                )
                
                if dist < self.patch_size * 2:
                    matches.append({
                        'target_idx': i,
                        'ref_idx': j,
                        'target_pos': target_pos,
                        'ref_pos': ref_pos,
                        'distance': distances[i, j],
                        'spatial_dist': dist,
                        'scale_diff': abs(
                            target_data['scales'][i] -
                            reference_data['scales'][j]
                        )
                    })
        
        return matches
    
    def _apply_attention(
        self,
        matches: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Apply attention mechanism to weight matches by reliability
        """
        if not matches:
            return []
        
        # Extract features for attention
        features = []
        for match in matches:
            feat = np.array([
                1.0 / (match['distance'] + 1e-6),
                1.0 / (match['spatial_dist'] + 1e-6),
                1.0 / (match['scale_diff'] + 1.0)
            ])
            features.append(feat)
        
        features = np.array(features)
        
        # Simple self-attention
        # Normalize features
        features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-6)
        
        # Compute attention scores
        attention_scores = features @ features.T
        attention_weights = np.exp(attention_scores) / np.sum(
            np.exp(attention_scores), axis=1, keepdims=True
        )
        
        # Apply attention to get weighted features
        weighted_features = attention_weights @ features
        
        # Compute final weights
        weights = np.mean(weighted_features, axis=1)
        weights = weights / np.sum(weights)
        
        # Add weights to matches
        for i, match in enumerate(matches):
            match['weight'] = weights[i]
        
        # Filter out low-weight matches
        weighted_matches = [
            m for m in matches if m['weight'] > 1.0 / len(matches)
        ]
        
        return weighted_matches
    
    def _estimate_transform(
        self,
        matches: List[Dict[str, Any]]
    ) -> Transform:
        """
        Estimate transform from weighted matches
        """
        if len(matches) < 4:
            return Transform()  # Identity if not enough matches
        
        # Extract point correspondences
        src_points = []
        dst_points = []
        weights = []
        
        for match in matches:
            src_points.append(match['ref_pos'])
            dst_points.append(match['target_pos'])
            weights.append(match['weight'])
        
        src_points = np.array(src_points)
        dst_points = np.array(dst_points)
        weights = np.array(weights)
        
        # Weighted least squares for transform estimation
        # Center points
        src_center = np.average(src_points, axis=0, weights=weights)
        dst_center = np.average(dst_points, axis=0, weights=weights)
        
        src_centered = src_points - src_center
        dst_centered = dst_points - dst_center
        
        # Apply weights
        src_weighted = src_centered * weights[:, np.newaxis]
        dst_weighted = dst_centered * weights[:, np.newaxis]
        
        # Estimate rotation and scale using SVD
        H = src_weighted.T @ dst_weighted
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Ensure proper rotation
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Extract scale
        scale = np.sqrt(np.sum(S) / np.sum(weights))
        
        # Extract rotation angle
        angle = np.arctan2(R[1, 0], R[0, 0]) * 180 / np.pi
        
        # Compute translation
        translation = dst_center - scale * R @ src_center
        
        # Create transform
        transform = Transform()
        transform.set_translation(translation[0], translation[1])
        transform.set_rotation(angle)
        transform.set_scale(scale)
        
        return transform
    
    def _downsample(
        self,
        image: np.ndarray,
        factor: int
    ) -> np.ndarray:
        """
        Downsample image by factor
        """
        # Anti-alias filter
        from scipy import ndimage
        sigma = factor / 2.0
        filtered = ndimage.gaussian_filter(image, sigma)
        
        # Subsample
        return filtered[::factor, ::factor]