"""
Main alignment pipeline orchestrating all components
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from tqdm import tqdm

from ..algorithms.fourier_align import FourierAligner
from ..algorithms.attention_matcher import AttentionMatcher
from ..algorithms.bundle_adjustment import BundleAdjuster
from ..algorithms.distortion import DistortionEstimator
from ..models.image import AlignedImage, ImageMetadata
from ..models.transform import Transform, Homography
from ..utils.canvas import CanvasManager
from ..utils.pyramid import ImagePyramid


logger = logging.getLogger(__name__)


class AlignmentPipeline:
    """
    Main pipeline for image alignment with global bundle adjustment
    """
    
    def __init__(
        self,
        overlap_ratio: float = 0.5,
        num_threads: Optional[int] = None,
        debug: bool = False,
        cache_dir: Optional[Path] = None
    ):
        self.overlap_ratio = overlap_ratio
        self.num_threads = num_threads
        self.debug = debug
        self.cache_dir = cache_dir
        
        # Initialize components
        self.fourier_aligner = FourierAligner()
        self.attention_matcher = AttentionMatcher()
        self.bundle_adjuster = BundleAdjuster()
        self.distortion_estimator = DistortionEstimator()
        self.canvas_manager = CanvasManager()
        
    def align(
        self,
        images: List[np.ndarray],
        metadata: List[ImageMetadata]
    ) -> Dict[str, Any]:
        """
        Main alignment method
        """
        logger.info(f"Starting alignment of {len(images)} images")
        
        # Step 1: Build image pyramids for multi-resolution processing
        logger.info("Building image pyramids...")
        pyramids = []
        for img in tqdm(images, desc="Building pyramids"):
            pyramid = ImagePyramid(img, levels=5)
            pyramids.append(pyramid)
        
        # Step 2: Initial pairwise alignment using Fourier methods
        logger.info("Performing initial Fourier-based alignment...")
        initial_transforms = self._initial_alignment(pyramids)
        
        # Step 3: Refine with attention-based matching
        logger.info("Refining with attention-based matching...")
        refined_transforms = self._attention_refinement(
            pyramids, initial_transforms
        )
        
        # Step 4: Estimate distortion parameters
        logger.info("Estimating distortion parameters...")
        distortion_params = self._estimate_distortion(
            pyramids, refined_transforms
        )
        
        # Step 5: Global bundle adjustment
        logger.info("Performing global bundle adjustment...")
        global_transforms = self._bundle_adjustment(
            pyramids, refined_transforms, distortion_params
        )
        
        # Step 6: Final refinement at full resolution
        logger.info("Final refinement at full resolution...")
        final_transforms = self._final_refinement(
            images, global_transforms, distortion_params
        )
        
        # Step 7: Compute canvas size and warp images
        logger.info("Computing canvas and warping images...")
        canvas_size = self.canvas_manager.compute_canvas_size(
            images, final_transforms
        )
        
        aligned_images = []
        masks = []
        
        for i, (img, transform) in enumerate(zip(images, final_transforms)):
            warped, mask = self._warp_image(
                img, transform, distortion_params[i], canvas_size
            )
            aligned_images.append(warped)
            masks.append(mask)
        
        # Compute metrics
        metrics = self._compute_metrics(
            images, final_transforms, aligned_images
        )
        
        return {
            'aligned_images': aligned_images,
            'masks': masks,
            'transforms': final_transforms,
            'distortion_params': distortion_params,
            'canvas_size': canvas_size,
            'metrics': metrics
        }
    
    def _initial_alignment(
        self,
        pyramids: List[ImagePyramid]
    ) -> List[Transform]:
        """
        Initial alignment using Fourier phase correlation
        """
        transforms = [Transform() for _ in pyramids]  # Identity for first image
        
        # Align sequential pairs
        for i in range(1, len(pyramids)):
            # Start at coarsest level
            for level in range(pyramids[0].num_levels - 1, -1, -1):
                prev_img = pyramids[i-1].get_level(level)
                curr_img = pyramids[i].get_level(level)
                
                # Estimate transform at this level
                transform = self.fourier_aligner.align(prev_img, curr_img)
                
                # Scale up for next level
                if level > 0:
                    transform = transform.scale(2.0)
                
                # Accumulate transform
                transforms[i] = transforms[i].compose(transform)
        
        return transforms
    
    def _attention_refinement(
        self,
        pyramids: List[ImagePyramid],
        initial_transforms: List[Transform]
    ) -> List[Transform]:
        """
        Refine alignment using attention-based patch matching
        """
        refined_transforms = initial_transforms.copy()
        
        # Work on medium resolution for efficiency
        level = min(2, pyramids[0].num_levels - 1)
        
        for i in range(len(pyramids)):
            if i == 0:
                continue
                
            # Find overlapping neighbors
            neighbors = self._find_neighbors(i, len(pyramids))
            
            # Collect patches from neighbors
            patches = []
            for j in neighbors:
                img = pyramids[j].get_level(level)
                patch_set = self.attention_matcher.extract_patches(img)
                patches.append((j, patch_set))
            
            # Match and refine
            curr_img = pyramids[i].get_level(level)
            refinement = self.attention_matcher.match_and_refine(
                curr_img,
                patches,
                initial_transforms[i]
            )
            
            refined_transforms[i] = refinement
        
        return refined_transforms
    
    def _estimate_distortion(
        self,
        pyramids: List[ImagePyramid],
        transforms: List[Transform]
    ) -> List[Dict[str, Any]]:
        """
        Estimate lens distortion parameters
        """
        distortion_params = []
        
        for i, pyramid in enumerate(pyramids):
            # Use medium resolution for efficiency
            img = pyramid.get_level(2)
            
            # Estimate distortion
            params = self.distortion_estimator.estimate(
                img,
                transforms[i]
            )
            
            distortion_params.append(params)
        
        return distortion_params
    
    def _bundle_adjustment(
        self,
        pyramids: List[ImagePyramid],
        transforms: List[Transform],
        distortion_params: List[Dict[str, Any]]
    ) -> List[Transform]:
        """
        Global bundle adjustment across all images
        """
        # Extract features for bundle adjustment
        all_features = []
        
        level = 2  # Medium resolution
        for pyramid in pyramids:
            img = pyramid.get_level(level)
            features = self.bundle_adjuster.extract_features(img)
            all_features.append(features)
        
        # Run global optimization
        optimized_transforms = self.bundle_adjuster.optimize(
            all_features,
            transforms,
            distortion_params,
            self.overlap_ratio
        )
        
        return optimized_transforms
    
    def _final_refinement(
        self,
        images: List[np.ndarray],
        transforms: List[Transform],
        distortion_params: List[Dict[str, Any]]
    ) -> List[Transform]:
        """
        Final refinement at full resolution
        """
        final_transforms = transforms.copy()
        
        # Refine each transform with sub-pixel accuracy
        for i in range(1, len(images)):
            # Apply current transform and distortion
            warped_prev = self._apply_transform(
                images[i-1],
                final_transforms[i-1],
                distortion_params[i-1]
            )
            
            warped_curr = self._apply_transform(
                images[i],
                final_transforms[i],
                distortion_params[i]
            )
            
            # Fine-tune alignment
            refinement = self.fourier_aligner.subpixel_refine(
                warped_prev,
                warped_curr
            )
            
            final_transforms[i] = final_transforms[i].compose(refinement)
        
        return final_transforms
    
    def _warp_image(
        self,
        image: np.ndarray,
        transform: Transform,
        distortion_params: Dict[str, Any],
        canvas_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Warp image to canvas with distortion correction
        """
        # First correct distortion
        undistorted = self.distortion_estimator.undistort(
            image,
            distortion_params
        )
        
        # Then apply homography
        warped, mask = transform.apply_to_image(
            undistorted,
            canvas_size
        )
        
        return warped, mask
    
    def _apply_transform(
        self,
        image: np.ndarray,
        transform: Transform,
        distortion_params: Dict[str, Any]
    ) -> np.ndarray:
        """
        Apply transform and distortion correction
        """
        undistorted = self.distortion_estimator.undistort(
            image,
            distortion_params
        )
        
        warped, _ = transform.apply_to_image(
            undistorted,
            image.shape[:2]
        )
        
        return warped
    
    def _find_neighbors(
        self,
        index: int,
        total: int
    ) -> List[int]:
        """
        Find neighboring images based on expected overlap
        """
        neighbors = []
        
        # Simple sequential assumption for now
        if index > 0:
            neighbors.append(index - 1)
        if index < total - 1:
            neighbors.append(index + 1)
        
        # Could be extended to handle 2D grids
        
        return neighbors
    
    def _compute_metrics(
        self,
        original_images: List[np.ndarray],
        transforms: List[Transform],
        aligned_images: List[np.ndarray]
    ) -> Dict[str, float]:
        """
        Compute alignment quality metrics
        """
        errors = []
        
        for i in range(1, len(aligned_images)):
            # Compute reprojection error in overlap regions
            overlap_mask = self._compute_overlap_mask(
                aligned_images[i-1],
                aligned_images[i]
            )
            
            if np.any(overlap_mask):
                diff = np.abs(
                    aligned_images[i-1][overlap_mask] -
                    aligned_images[i][overlap_mask]
                )
                error = np.mean(diff)
                errors.append(error)
        
        return {
            'avg_error': np.mean(errors) if errors else 0.0,
            'max_error': np.max(errors) if errors else 0.0,
            'std_error': np.std(errors) if errors else 0.0
        }
    
    def _compute_overlap_mask(
        self,
        img1: np.ndarray,
        img2: np.ndarray
    ) -> np.ndarray:
        """
        Compute mask of overlapping regions
        """
        mask1 = img1[..., 0] > 0 if img1.ndim == 3 else img1 > 0
        mask2 = img2[..., 0] > 0 if img2.ndim == 3 else img2 > 0
        
        return mask1 & mask2