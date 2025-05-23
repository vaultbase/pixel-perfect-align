# Usage Guide

## Installation

```bash
# Clone the repository
git clone https://github.com/vaultbase/pixel-perfect-align.git
cd pixel-perfect-align

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

## Basic Usage

### Command Line

```bash
# Basic alignment
python align.py --input-dir /path/to/images --output-dir /path/to/output

# With transform export
python align.py --input-dir /path/to/images --output-dir /path/to/output --export-transforms

# Debug mode with intermediate outputs
python align.py --input-dir /path/to/images --output-dir /path/to/output --debug
```

### Python API

```python
from src.core.pipeline import AlignmentPipeline
from src.utils.io import ImageLoader, ResultExporter

# Load images
loader = ImageLoader()
images = []
metadata = []

for path in image_paths:
    img, meta = loader.load_image(path)
    images.append(img)
    metadata.append(meta)

# Create pipeline
pipeline = AlignmentPipeline(overlap_ratio=0.5)

# Run alignment
results = pipeline.align(images, metadata)

# Export results
exporter = ResultExporter(output_dir)
exporter.export_aligned_images(
    results['aligned_images'],
    results['canvas_size'],
    metadata
)
```

## Advanced Options

### Performance Tuning

```bash
# Limit resolution for faster initial alignment
python align.py --input-dir ./images --output-dir ./output --max-resolution 4000

# Use multiple threads
python align.py --input-dir ./images --output-dir ./output --num-threads 8
```

### Image Requirements

- **Format**: TIFF, JPEG, PNG, RAW (DNG, RAF, NEF, CR2, ARW)
- **Overlap**: 30-70% overlap between adjacent images
- **Resolution**: Optimized for 50-100MP images
- **Order**: Images should be in sequential capture order

### Output Files

1. **aligned_XXX_filename.tif**: Individual aligned images (16-bit TIFF)
2. **composite.tif**: Blended composite of all images
3. **transforms.json**: Transformation parameters for each image
4. **summary.json**: Alignment metrics and statistics
5. **alignment.log**: Detailed processing log

## Troubleshooting

### Out of Memory

If processing very large images:
1. Use `--max-resolution` to limit initial processing resolution
2. Reduce number of pyramid levels in the code
3. Process in smaller batches

### Poor Alignment

If alignment quality is poor:
1. Ensure images have sufficient overlap (>40%)
2. Check that images are in correct sequential order
3. Use `--debug` mode to see intermediate results
4. Verify images have sufficient features for matching

### Slow Performance

To improve speed:
1. Use `--num-threads` to enable parallel processing
2. Lower `--max-resolution` for initial alignment
3. Ensure SSD storage for cache directory
4. Close other memory-intensive applications