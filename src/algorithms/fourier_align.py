"""
Fourier-based alignment using phase correlation
"""

import numpy as np
from scipy import ndimage
from scipy.fft import fft2, ifft2, fftshift
from typing import Tuple, Optional

from ..models.transform import Transform


class FourierAligner:
    """
    Implements Fourier-based alignment with phase correlation
    """
    
    def __init__(self, upsample_factor: int = 10):
        self.upsample_factor = upsample_factor
    
    def align(
        self,
        reference: np.ndarray,
        target: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Transform:
        """
        Align target image to reference using phase correlation
        """
        # Convert to grayscale if needed
        if reference.ndim == 3:
            reference = np.mean(reference, axis=2)
        if target.ndim == 3:
            target = np.mean(target, axis=2)
        
        # Apply window to reduce edge effects
        window = self._create_window(reference.shape)
        ref_windowed = reference * window
        tgt_windowed = target * window
        
        # Phase correlation for translation
        shift = self._phase_correlation(ref_windowed, tgt_windowed)
        
        # Log-polar transform for rotation and scale
        ref_logpolar = self._logpolar_transform(ref_windowed)
        tgt_logpolar = self._logpolar_transform(tgt_windowed)
        
        # Phase correlation in log-polar space
        logpolar_shift = self._phase_correlation(ref_logpolar, tgt_logpolar)
        
        # Convert to rotation and scale
        rotation = -logpolar_shift[1] * 180.0 / ref_logpolar.shape[1]
        scale = np.exp(logpolar_shift[0] / ref_logpolar.shape[0] * np.log(min(reference.shape) / 2))
        
        # Create transform
        transform = Transform()
        transform.set_translation(shift[1], shift[0])
        transform.set_rotation(rotation)
        transform.set_scale(scale)
        
        return transform
    
    def subpixel_refine(
        self,
        reference: np.ndarray,
        target: np.ndarray,
        initial_shift: Optional[Tuple[float, float]] = None
    ) -> Transform:
        """
        Refine alignment to sub-pixel accuracy
        """
        if reference.ndim == 3:
            reference = np.mean(reference, axis=2)
        if target.ndim == 3:
            target = np.mean(target, axis=2)
        
        # Compute cross-power spectrum
        ref_fft = fft2(reference)
        tgt_fft = fft2(target)
        cross_power = ref_fft * np.conj(tgt_fft)
        cross_power /= np.abs(cross_power) + 1e-10
        
        # Initial shift from cross-correlation peak
        correlation = np.real(ifft2(cross_power))
        if initial_shift is None:
            peak = np.unravel_index(np.argmax(correlation), correlation.shape)
            initial_shift = (
                peak[0] - reference.shape[0] // 2 if peak[0] > reference.shape[0] // 2 else peak[0],
                peak[1] - reference.shape[1] // 2 if peak[1] > reference.shape[1] // 2 else peak[1]
            )
        
        # Upsampled DFT around peak
        shift = self._upsampled_dft(
            cross_power,
            self.upsample_factor,
            initial_shift
        )
        
        transform = Transform()
        transform.set_translation(shift[1], shift[0])
        
        return transform
    
    def _phase_correlation(
        self,
        img1: np.ndarray,
        img2: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute phase correlation between two images
        """
        # Compute FFTs
        f1 = fft2(img1)
        f2 = fft2(img2)
        
        # Cross-power spectrum
        cross_power = f1 * np.conj(f2)
        cross_power /= np.abs(cross_power) + 1e-10
        
        # Inverse FFT to get correlation
        correlation = np.real(ifft2(cross_power))
        
        # Find peak
        peak = np.unravel_index(np.argmax(correlation), correlation.shape)
        
        # Convert to shift
        shift_y = peak[0] - img1.shape[0] // 2 if peak[0] > img1.shape[0] // 2 else peak[0]
        shift_x = peak[1] - img1.shape[1] // 2 if peak[1] > img1.shape[1] // 2 else peak[1]
        
        return (shift_y, shift_x)
    
    def _logpolar_transform(
        self,
        img: np.ndarray,
        angles: int = 360,
        radii: int = None
    ) -> np.ndarray:
        """
        Convert image to log-polar coordinates
        """
        if radii is None:
            radii = min(img.shape) // 2
        
        center = (img.shape[1] // 2, img.shape[0] // 2)
        max_radius = min(center)
        
        # Create log-polar grid
        theta = np.linspace(0, 2 * np.pi, angles, endpoint=False)
        log_r = np.linspace(0, np.log(max_radius), radii)
        r = np.exp(log_r)
        
        # Convert to Cartesian coordinates
        r_grid, theta_grid = np.meshgrid(r, theta)
        x = r_grid * np.cos(theta_grid) + center[0]
        y = r_grid * np.sin(theta_grid) + center[1]
        
        # Interpolate
        coords = np.array([y.ravel(), x.ravel()])
        logpolar = ndimage.map_coordinates(
            img,
            coords,
            order=3,
            mode='constant'
        ).reshape(angles, radii)
        
        return logpolar
    
    def _create_window(
        self,
        shape: Tuple[int, int],
        window_type: str = 'hann'
    ) -> np.ndarray:
        """
        Create 2D window function
        """
        if window_type == 'hann':
            window_y = np.hanning(shape[0])
            window_x = np.hanning(shape[1])
        elif window_type == 'hamming':
            window_y = np.hamming(shape[0])
            window_x = np.hamming(shape[1])
        else:
            window_y = np.ones(shape[0])
            window_x = np.ones(shape[1])
        
        window_2d = window_y[:, np.newaxis] * window_x[np.newaxis, :]
        
        return window_2d
    
    def _upsampled_dft(
        self,
        cross_power: np.ndarray,
        upsample_factor: int,
        initial_shift: Tuple[float, float]
    ) -> Tuple[float, float]:
        """
        Compute upsampled DFT around initial shift estimate
        """
        # Size of upsampled region
        upsampled_size = int(np.ceil(upsample_factor * 1.5))
        
        # Frequency ranges
        row_shift = initial_shift[0]
        col_shift = initial_shift[1]
        
        # Create upsampled grid around initial shift
        row_kernel = np.arange(-upsampled_size, upsampled_size + 1) / upsample_factor
        col_kernel = np.arange(-upsampled_size, upsampled_size + 1) / upsample_factor
        
        row_kernel += row_shift
        col_kernel += col_shift
        
        # Compute upsampled correlation
        kernr = np.exp(-2j * np.pi * row_kernel[:, np.newaxis] * np.fft.fftfreq(cross_power.shape[0])[:, np.newaxis])
        kernc = np.exp(-2j * np.pi * col_kernel[:, np.newaxis] * np.fft.fftfreq(cross_power.shape[1])[:, np.newaxis])
        
        upsampled = kernr.T @ cross_power @ kernc
        
        # Find peak in upsampled correlation
        peak = np.unravel_index(np.argmax(np.abs(upsampled)), upsampled.shape)
        
        # Convert back to shift
        refined_row = row_kernel[peak[0]]
        refined_col = col_kernel[peak[1]]
        
        return (refined_row, refined_col)