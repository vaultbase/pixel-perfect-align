#!/usr/bin/env python3
"""
Simple alignment script with minimal dependencies
Just drag a folder onto this script or run: python simple_align.py /path/to/folder
"""

import sys
import os
from pathlib import Path
import logging

# Set up simple logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def find_images(folder_path):
    """Find all valid image files in folder"""
    valid_extensions = {'.tif', '.tiff', '.jpg', '.jpeg', '.png'}
    images = []
    
    for ext in valid_extensions:
        images.extend(folder_path.glob(f"*{ext}"))
        images.extend(folder_path.glob(f"*{ext.upper()}"))
    
    # Filter out hidden files
    images = [img for img in images if not img.name.startswith('.') and not img.name.startswith('._')]
    
    return sorted(set(images))


def main():
    if len(sys.argv) < 2:
        print("\n🖼️  Pixel Perfect Align - Simple Mode")
        print("\nUsage:")
        print("  python simple_align.py /path/to/images")
        print("\nOr drag a folder onto this script!")
        print("\nOutput will be saved to 'Aligned' subfolder")
        return 1
    
    input_path = Path(sys.argv[1])
    
    if not input_path.exists():
        print(f"❌ Error: Path does not exist: {input_path}")
        return 1
    
    if not input_path.is_dir():
        # If it's a file, use its parent directory
        input_path = input_path.parent
    
    print(f"\n🚀 Starting Pixel Perfect Align")
    print(f"📁 Input folder: {input_path}")
    
    # Find images
    images = find_images(input_path)
    print(f"🖼️  Found {len(images)} images")
    
    if len(images) < 2:
        print("❌ Need at least 2 images to align")
        return 1
    
    # Create output directory
    output_dir = input_path / "Aligned"
    output_dir.mkdir(exist_ok=True)
    print(f"📂 Output folder: {output_dir}")
    
    try:
        # Import only when needed to avoid early crashes
        from src.core.pipeline import AlignmentPipeline
        from src.utils.io import ImageLoader, ResultExporter
        
        print("\n⏳ Loading images...")
        loader = ImageLoader()
        loaded_images = []
        metadata = []
        
        for i, img_path in enumerate(images):
            print(f"   Loading {img_path.name} ({i+1}/{len(images)})")
            try:
                img, meta = loader.load_image(img_path)
                loaded_images.append(img)
                metadata.append(meta)
            except Exception as e:
                print(f"   ⚠️  Skipping {img_path.name}: {e}")
        
        if len(loaded_images) < 2:
            print("❌ Not enough valid images after loading")
            return 1
        
        print(f"\n🔄 Starting alignment of {len(loaded_images)} images...")
        
        # Create pipeline with safe defaults
        pipeline = AlignmentPipeline(
            overlap_ratio=0.5,
            debug=False
        )
        
        # Run alignment
        results = pipeline.align(loaded_images, metadata)
        
        print("\n💾 Saving results...")
        exporter = ResultExporter(output_dir)
        
        # Export aligned images
        exporter.export_aligned_images(
            results['aligned_images'],
            results['canvas_size'],
            metadata
        )
        
        # Export transforms
        exporter.export_transforms(results['transforms'])
        
        # Generate composite
        exporter.generate_composite(
            results['aligned_images'],
            results['masks']
        )
        
        print("\n✅ Alignment completed successfully!")
        print(f"📊 Average error: {results['metrics']['avg_error']:.3f} pixels")
        print(f"📁 Results saved to: {output_dir}")
        
        # Open output folder
        if sys.platform == 'darwin':
            os.system(f'open "{output_dir}"')
        
    except Exception as e:
        print(f"\n❌ Error during alignment: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())