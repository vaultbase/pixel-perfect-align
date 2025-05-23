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

### Simplest - Command Line
```bash
# Just run with a folder path!
python simple_align.py /path/to/images
```

### GUI Options

#### Tkinter GUI (Most Stable)
```bash
python tk_gui.py
```
Simple, stable GUI that works well on all platforms.

#### PyQt6 GUI (Feature-rich but may have issues on macOS)
```bash
python align_gui.py
```

### Batch Processing
```bash
# Process multiple folders at once
./batch_align.sh folder1 folder2 folder3

# Or drag multiple folders onto the script
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