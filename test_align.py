#!/usr/bin/env python3
"""
Test script for pixel perfect alignment
"""

import sys
from pathlib import Path
import numpy as np
import cv2
from src.core.pipeline import AlignmentPipeline
from src.utils.io import ImageLoader, ResultExporter
from src.models.image import ImageMetadata


def create_test_images():
    """Create synthetic test images with known transformations"""
    print("Creating test images...")
    
    # Create base pattern
    size = 1000
    pattern = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Add grid pattern
    for i in range(0, size, 50):
        pattern[i, :] = 255
        pattern[:, i] = 255
    
    # Add some features
    cv2.circle(pattern, (300, 300), 50, (255, 0, 0), -1)
    cv2.rectangle(pattern, (600, 600), (800, 800), (0, 255, 0), -1)
    cv2.putText(pattern, "TEST", (400, 500), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 10)
    
    # Create test directory
    test_dir = Path("test_images")
    test_dir.mkdir(exist_ok=True)
    
    # Generate transformed versions
    transforms = [
        {"tx": 0, "ty": 0, "angle": 0},
        {"tx": 400, "ty": 100, "angle": 5},
        {"tx": 300, "ty": 400, "angle": -3},
        {"tx": 100, "ty": 300, "angle": 2},
    ]
    
    for i, t in enumerate(transforms):
        # Create transformation matrix
        M = cv2.getRotationMatrix2D((size//2, size//2), t["angle"], 1.0)
        M[0, 2] += t["tx"]
        M[1, 2] += t["ty"]
        
        # Apply transform
        transformed = cv2.warpAffine(pattern, M, (size*2, size*2))
        
        # Save
        cv2.imwrite(str(test_dir / f"test_{i:02d}.png"), transformed)
    
    print(f"Created {len(transforms)} test images in {test_dir}")
    return test_dir


def main():
    # Create test images if needed
    test_dir = Path("test_images")
    if not test_dir.exists() or len(list(test_dir.glob("*.png"))) == 0:
        test_dir = create_test_images()
    
    # Setup output directory
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    # Load images
    print("\nLoading images...")
    loader = ImageLoader(max_resolution=2000)
    
    image_paths = sorted(test_dir.glob("*.png"))
    images = []
    metadata = []
    
    for path in image_paths:
        img, meta = loader.load_image(path)
        images.append(img)
        metadata.append(meta)
        print(f"Loaded: {path.name} ({meta.width}x{meta.height})")
    
    # Run alignment
    print("\nRunning alignment pipeline...")
    pipeline = AlignmentPipeline(
        overlap_ratio=0.5,
        debug=True
    )
    
    try:
        results = pipeline.align(images, metadata)
        
        # Export results
        print("\nExporting results...")
        exporter = ResultExporter(output_dir)
        
        # Export aligned images
        aligned_paths = exporter.export_aligned_images(
            results['aligned_images'],
            results['canvas_size'],
            metadata
        )
        
        # Export transforms
        transform_path = exporter.export_transforms(results['transforms'])
        print(f"Transforms saved to: {transform_path}")
        
        # Generate composite
        composite_path = exporter.generate_composite(
            results['aligned_images'],
            results['masks']
        )
        print(f"Composite saved to: {composite_path}")
        
        # Print metrics
        print("\nAlignment Metrics:")
        print(f"Canvas size: {results['canvas_size']}")
        print(f"Average error: {results['metrics']['avg_error']:.3f} pixels")
        print(f"Max error: {results['metrics']['max_error']:.3f} pixels")
        
    except Exception as e:
        print(f"Error during alignment: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())