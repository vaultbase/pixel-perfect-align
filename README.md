# Pixel Perfect Align

State-of-the-art image alignment system for high-resolution image mosaicing with global bundle adjustment, designed specifically for Fujifilm GFX100S II images.

## Features

- **Coarse-to-Fine Alignment**: Multi-resolution approach using Fourier decomposition
- **Global Bundle Adjustment**: Simultaneous optimization across all images
- **Advanced Distortion Correction**: Handles lens distortion and perspective warping
- **Attention-Based Patching**: Intelligent feature matching across overlapping regions
- **High Performance**: Optimized for MacBook Pro, completes in under 10 minutes
- **16-bit Linear TIFF Output**: Preserves full dynamic range

## Architecture

The system uses a hierarchical approach:
1. Initial rough alignment using phase correlation
2. Multi-scale Fourier decomposition for frequency-based alignment
3. Attention-based patch matching for fine details
4. Global optimization with distortion parameter estimation
5. Final homography refinement with sub-pixel accuracy

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python align.py --input-dir ./images --output-dir ./aligned --export-transforms
```

## Output

- Aligned 16-bit linear TIFF images on full canvas
- Transform parameters (JSON)
- Composite image using overlap averaging
- Distortion coefficients and homographies