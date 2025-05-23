# Pixel Perfect Align

State-of-the-art image alignment system for high-resolution image mosaicing with global bundle adjustment, designed specifically for Fujifilm GFX100S II images.

## Features

- **Zero-Effort GUI**: Simple drag-and-drop interface for effortless alignment
- **Automatic Output**: Creates "Aligned" folder automatically
- **Coarse-to-Fine Alignment**: Multi-resolution approach using Fourier decomposition
- **Global Bundle Adjustment**: Simultaneous optimization across all images
- **Advanced Distortion Correction**: Handles lens distortion and perspective warping
- **Attention-Based Patching**: Intelligent feature matching across overlapping regions
- **High Performance**: Optimized for MacBook Pro, completes in under 10 minutes
- **16-bit Linear TIFF Output**: Preserves full dynamic range

## Quick Start

### GUI Application (Recommended)
```bash
python align_gui.py
```
- Drag & drop images or browse for folder
- Click "Start Alignment" - that's it!

### Command Line - Easy Mode
```bash
# Just pass a folder - everything else is automatic!
python easy_align.py /path/to/images
```

### Command Line - Advanced
```bash
# Automatic output to input-dir/Aligned
python align.py --input-dir ./images

# Custom output directory
python align.py --input-dir ./images --output-dir ./custom-output
```

## Installation

```bash
pip install -r requirements.txt
```

## Output

All outputs are automatically saved to an "Aligned" subfolder:
- Aligned 16-bit linear TIFF images on full canvas
- Transform parameters (transforms.json)
- Composite image using overlap averaging
- Processing summary and metrics