"""
Image I/O utilities
"""

import json
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List

import numpy as np
import cv2
import tifffile
import rawpy
from PIL import Image
import imageio

from ..models.image import ImageMetadata
from ..models.transform import Transform, Homography


logger = logging.getLogger(__name__)


class ImageLoader:
    """
    Handles loading images from various formats
    """
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        max_resolution: Optional[int] = None
    ):
        self.cache_dir = cache_dir
        self.max_resolution = max_resolution
        self.supported_formats = {
            '.tif', '.tiff', '.jpg', '.jpeg', '.png',
            '.raw', '.dng', '.raf', '.nef', '.cr2', '.arw'
        }
    
    def load_image(
        self,
        path: Path,
        as_float: bool = True
    ) -> Tuple[np.ndarray, ImageMetadata]:
        """
        Load image and metadata
        """
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        
        # Skip hidden/system files
        if path.name.startswith('._') or path.name.startswith('.'):
            raise ValueError(f"Skipping hidden/system file: {path.name}")
        
        suffix = path.suffix.lower()
        
        try:
            if suffix in {'.raw', '.dng', '.raf', '.nef', '.cr2', '.arw'}:
                image, metadata = self._load_raw(path)
            elif suffix in {'.tif', '.tiff'}:
                image, metadata = self._load_tiff(path)
            else:
                image, metadata = self._load_standard(path)
        except Exception as e:
            logger.error(f"Failed to load {path.name}: {str(e)}")
            raise ValueError(f"Failed to load {path.name}: {str(e)}")
        
        # Resize if needed
        if self.max_resolution and max(image.shape[:2]) > self.max_resolution:
            image = self._resize_image(image)
            metadata.width = image.shape[1]
            metadata.height = image.shape[0]
        
        # Convert to float if requested
        if as_float and image.dtype != np.float32:
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            elif image.dtype == np.uint16:
                image = image.astype(np.float32) / 65535.0
        
        return image, metadata
    
    def _load_raw(
        self,
        path: Path
    ) -> Tuple[np.ndarray, ImageMetadata]:
        """
        Load RAW image using rawpy
        """
        with rawpy.imread(str(path)) as raw:
            # Process with default settings
            rgb = raw.postprocess(
                use_camera_wb=True,
                half_size=False,
                no_auto_bright=True,
                output_bps=16
            )
            
            # Get metadata
            metadata = ImageMetadata(
                filename=path.name,
                path=path,
                width=rgb.shape[1],
                height=rgb.shape[0],
                channels=3,
                bit_depth=16,
                color_space='sRGB'
            )
            
            # Extract EXIF if available
            try:
                metadata.exif_data = {
                    'iso': raw.iso_speed,
                    'shutter': raw.shutter_speed,
                    'aperture': raw.aperture,
                    'focal_length': raw.focal_length
                }
            except:
                pass
        
        return rgb, metadata
    
    def _load_tiff(
        self,
        path: Path
    ) -> Tuple[np.ndarray, ImageMetadata]:
        """
        Load TIFF image
        """
        image = tifffile.imread(str(path))
        
        # Handle different channel arrangements
        if image.ndim == 2:
            channels = 1
        elif image.ndim == 3:
            if image.shape[2] in [3, 4]:
                channels = image.shape[2]
            elif image.shape[0] in [3, 4]:
                # Channels first
                image = np.transpose(image, (1, 2, 0))
                channels = image.shape[2]
            else:
                channels = image.shape[2]
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")
        
        # Determine bit depth
        if image.dtype == np.uint8:
            bit_depth = 8
        elif image.dtype == np.uint16:
            bit_depth = 16
        elif image.dtype == np.float32:
            bit_depth = 32
        else:
            bit_depth = 8
        
        metadata = ImageMetadata(
            filename=path.name,
            path=path,
            width=image.shape[1],
            height=image.shape[0],
            channels=channels,
            bit_depth=bit_depth,
            color_space='sRGB'
        )
        
        return image, metadata
    
    def _load_standard(
        self,
        path: Path
    ) -> Tuple[np.ndarray, ImageMetadata]:
        """
        Load standard image formats (JPEG, PNG)
        """
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        
        if image is None:
            raise ValueError(f"Failed to load image: {path}")
        
        # Convert BGR to RGB
        if image.ndim == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        
        # Determine bit depth
        if image.dtype == np.uint8:
            bit_depth = 8
        elif image.dtype == np.uint16:
            bit_depth = 16
        else:
            bit_depth = 8
        
        metadata = ImageMetadata(
            filename=path.name,
            path=path,
            width=image.shape[1],
            height=image.shape[0],
            channels=image.shape[2] if image.ndim == 3 else 1,
            bit_depth=bit_depth,
            color_space='sRGB'
        )
        
        return image, metadata
    
    def _resize_image(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        """
        Resize image to max resolution
        """
        h, w = image.shape[:2]
        scale = self.max_resolution / max(h, w)
        
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        return cv2.resize(
            image,
            (new_w, new_h),
            interpolation=cv2.INTER_CUBIC
        )


class ResultExporter:
    """
    Handles exporting alignment results
    """
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_aligned_images(
        self,
        aligned_images: List[np.ndarray],
        canvas_size: Tuple[int, int],
        metadata: List[ImageMetadata]
    ) -> List[Path]:
        """
        Export aligned images as 16-bit linear TIFF
        """
        paths = []
        
        for i, (image, meta) in enumerate(zip(aligned_images, metadata)):
            # Convert to 16-bit
            if image.dtype == np.float32 or image.dtype == np.float64:
                # Assume normalized 0-1
                image_16bit = (image * 65535).astype(np.uint16)
            elif image.dtype == np.uint8:
                image_16bit = (image.astype(np.float32) / 255.0 * 65535).astype(np.uint16)
            else:
                image_16bit = image.astype(np.uint16)
            
            # Save as TIFF
            filename = f"aligned_{i:03d}_{meta.filename.replace(meta.path.suffix, '.tif')}"
            output_path = self.output_dir / filename
            
            tifffile.imwrite(
                str(output_path),
                image_16bit,
                photometric='rgb' if image_16bit.ndim == 3 else 'minisblack',
                compression='none',
                metadata={
                    'Software': 'Pixel Perfect Align',
                    'ImageDescription': f'Aligned image {i} of {len(aligned_images)}'
                }
            )
            
            paths.append(output_path)
            logger.info(f"Exported: {output_path}")
        
        return paths
    
    def export_transforms(
        self,
        transforms: List[Transform]
    ) -> Path:
        """
        Export transformation parameters to JSON
        """
        data = {
            'version': '1.0',
            'num_images': len(transforms),
            'transforms': [t.to_dict() for t in transforms]
        }
        
        output_path = self.output_dir / 'transforms.json'
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return output_path
    
    def generate_composite(
        self,
        aligned_images: List[np.ndarray],
        masks: List[np.ndarray]
    ) -> Path:
        """
        Generate composite image using overlap averaging
        """
        if not aligned_images:
            raise ValueError("No images to composite")
        
        # Initialize accumulator and weight map
        h, w = aligned_images[0].shape[:2]
        if aligned_images[0].ndim == 3:
            accumulator = np.zeros((h, w, aligned_images[0].shape[2]), dtype=np.float64)
        else:
            accumulator = np.zeros((h, w), dtype=np.float64)
        
        weight_map = np.zeros((h, w), dtype=np.float64)
        
        # Accumulate weighted images
        for image, mask in zip(aligned_images, masks):
            # Create weight from mask with feathering
            weight = mask.astype(np.float32) / 255.0
            
            # Apply distance transform for smooth blending
            if np.any(weight > 0):
                dist = cv2.distanceTransform(
                    weight.astype(np.uint8),
                    cv2.DIST_L2,
                    5
                )
                weight = np.minimum(dist / 50.0, 1.0)
            
            # Accumulate
            if image.ndim == 3:
                for c in range(image.shape[2]):
                    accumulator[:, :, c] += image[:, :, c] * weight
            else:
                accumulator += image * weight
            
            weight_map += weight
        
        # Normalize by weights
        valid_mask = weight_map > 0
        if accumulator.ndim == 3:
            for c in range(accumulator.shape[2]):
                accumulator[valid_mask, c] /= weight_map[valid_mask]
        else:
            accumulator[valid_mask] /= weight_map[valid_mask]
        
        # Convert to 16-bit
        composite_16bit = (accumulator * 65535).astype(np.uint16)
        
        # Save
        output_path = self.output_dir / 'composite.tif'
        
        tifffile.imwrite(
            str(output_path),
            composite_16bit,
            photometric='rgb' if composite_16bit.ndim == 3 else 'minisblack',
            compression='none',
            metadata={
                'Software': 'Pixel Perfect Align',
                'ImageDescription': f'Composite of {len(aligned_images)} images'
            }
        )
        
        return output_path