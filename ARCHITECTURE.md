# Architecture Overview

## System Design

Pixel Perfect Align uses a multi-stage pipeline for achieving sub-pixel accurate image alignment:

### 1. Multi-Resolution Processing
- **Image Pyramids**: Gaussian pyramids for coarse-to-fine alignment
- **Fourier Decomposition**: Frequency-based alignment starting from low frequencies
- **Progressive Refinement**: Each level refines the previous alignment

### 2. Core Alignment Stages

#### Stage 1: Initial Fourier Alignment
- Phase correlation for translation estimation
- Log-polar transform for rotation and scale
- Operates on downsampled images for speed

#### Stage 2: Attention-Based Refinement
- Multi-scale patch extraction
- Attention mechanism for reliability weighting
- Robust to repetitive patterns

#### Stage 3: Distortion Estimation
- Detects straight lines for distortion analysis
- Estimates radial and tangential distortion coefficients
- Per-image distortion parameters

#### Stage 4: Global Bundle Adjustment
- Simultaneous optimization of all image parameters
- Feature tracking across multiple images
- Energy minimization with regularization

#### Stage 5: Final Refinement
- Full-resolution processing
- Sub-pixel accuracy using upsampled DFT
- Final homography computation

### 3. Key Components

```
src/
├── core/
│   └── pipeline.py          # Main orchestration
├── algorithms/
│   ├── fourier_align.py     # FFT-based alignment
│   ├── attention_matcher.py # Attention-based matching
│   ├── bundle_adjustment.py # Global optimization
│   └── distortion.py        # Lens distortion correction
├── models/
│   ├── transform.py         # Transform representations
│   └── image.py             # Image data models
└── utils/
    ├── io.py                # I/O operations
    ├── pyramid.py           # Multi-resolution support
    ├── canvas.py            # Canvas management
    └── logging.py           # Logging utilities
```

## Algorithm Details

### Fourier Phase Correlation
- Handles large translations efficiently
- Robust to illumination changes
- Sub-pixel refinement using matrix-multiply DFT

### Attention Mechanism
- Self-attention on patch descriptors
- Weights matches by reliability
- Handles occlusions and outliers

### Bundle Adjustment
- Levenberg-Marquardt optimization
- Robust cost function (Huber loss)
- Constraint satisfaction for global consistency

### Distortion Model
- Brown-Conrady model (k1, k2, k3, p1, p2)
- Automatic estimation from line straightness
- Per-image calibration

## Performance Optimizations

1. **Multi-threading**: Parallel processing where possible
2. **Memory Efficiency**: Streaming large images
3. **Caching**: Intermediate results cached to disk
4. **Numba JIT**: Critical loops accelerated
5. **Hierarchical Processing**: Reduces computation at each level

## Output Pipeline

1. **Aligned Images**: 16-bit linear TIFF with full dynamic range
2. **Transform Data**: JSON export of all parameters
3. **Composite Generation**: Feathered blending with distance weighting
4. **Quality Metrics**: Reprojection error and overlap statistics