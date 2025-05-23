#!/usr/bin/env python3
"""
Pixel Perfect Align - State-of-the-art image alignment system
Main entry point for the alignment pipeline
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from tqdm import tqdm

from src.core.pipeline import AlignmentPipeline
from src.utils.io import ImageLoader, ResultExporter
from src.utils.logging import setup_logging


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Pixel Perfect Align - Advanced image alignment with global bundle adjustment"
    )
    
    parser.add_argument(
        "--input-dir", 
        type=Path, 
        required=True,
        help="Directory containing input images"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=False,
        default=None,
        help="Directory for output aligned images (default: input-dir/Aligned)"
    )
    
    parser.add_argument(
        "--export-transforms",
        action="store_true",
        default=True,
        help="Export transformation parameters to JSON (default: True)"
    )
    
    parser.add_argument(
        "--composite",
        action="store_true",
        default=True,
        help="Generate composite image from aligned results"
    )
    
    parser.add_argument(
        "--max-resolution",
        type=int,
        default=None,
        help="Maximum resolution for initial alignment (None for full res)"
    )
    
    parser.add_argument(
        "--overlap-ratio",
        type=float,
        default=0.5,
        help="Expected overlap ratio between images (0.3-0.7)"
    )
    
    parser.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help="Number of threads to use (None for auto)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging and intermediate outputs"
    )
    
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("./cache"),
        help="Directory for caching intermediate results"
    )
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # Auto-create output directory if not specified
    if args.output_dir is None:
        args.output_dir = args.input_dir / "Aligned"
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    
    # Create output directory first for logging
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(log_level, args.output_dir / "alignment.log")
    
    logger.info("Pixel Perfect Align - Starting alignment pipeline")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Create cache directory
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Load images
    logger.info("Loading images...")
    loader = ImageLoader(
        cache_dir=args.cache_dir,
        max_resolution=args.max_resolution
    )
    
    image_paths = sorted(args.input_dir.glob("*"))
    image_paths = [p for p in image_paths if p.suffix.lower() in {'.tif', '.tiff', '.jpg', '.jpeg', '.png', '.dng', '.raw'}]
    
    if len(image_paths) < 2:
        logger.error("Need at least 2 images for alignment")
        sys.exit(1)
    
    logger.info(f"Found {len(image_paths)} images")
    
    images = []
    metadata = []
    
    for path in tqdm(image_paths, desc="Loading images"):
        img, meta = loader.load_image(path)
        images.append(img)
        metadata.append(meta)
    
    # Initialize alignment pipeline
    logger.info("Initializing alignment pipeline...")
    pipeline = AlignmentPipeline(
        overlap_ratio=args.overlap_ratio,
        num_threads=args.num_threads,
        debug=args.debug,
        cache_dir=args.cache_dir
    )
    
    # Run alignment
    logger.info("Running alignment...")
    results = pipeline.align(images, metadata)
    
    # Export results
    logger.info("Exporting results...")
    exporter = ResultExporter(args.output_dir)
    
    # Export aligned images
    aligned_paths = exporter.export_aligned_images(
        results['aligned_images'],
        results['canvas_size'],
        metadata
    )
    
    # Export transforms if requested
    if args.export_transforms:
        transform_path = exporter.export_transforms(results['transforms'])
        logger.info(f"Transforms exported to: {transform_path}")
    
    # Generate composite if requested
    if args.composite:
        composite_path = exporter.generate_composite(
            results['aligned_images'],
            results['masks']
        )
        logger.info(f"Composite generated: {composite_path}")
    
    # Summary
    logger.info("Alignment complete!")
    logger.info(f"Canvas size: {results['canvas_size']}")
    logger.info(f"Average alignment error: {results['metrics']['avg_error']:.3f} pixels")
    logger.info(f"Max alignment error: {results['metrics']['max_error']:.3f} pixels")
    
    # Save summary
    summary = {
        'canvas_size': results['canvas_size'],
        'num_images': len(images),
        'metrics': results['metrics'],
        'aligned_images': [str(p) for p in aligned_paths]
    }
    
    with open(args.output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()