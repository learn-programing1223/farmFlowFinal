"""
Edge Case Handler for Robust Field Performance
Handles challenging real-world scenarios that can cause failures
"""

import numpy as np
import cv2
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class EdgeCaseHandler:
    """
    Handles edge cases common in field photography:
    - Completely blown out images
    - Extremely dark images
    - Motion blur
    - Mixed lighting
    - Flash artifacts
    - Thermal noise (from overheated phone)
    """
    
    def __init__(self):
        """Initialize edge case handler"""
        self.min_usable_pixels = 0.1  # At least 10% pixels must be usable
        self.max_blur_score = 100  # Laplacian variance threshold for blur
        
    def handle_edge_cases(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Main method to detect and handle edge cases
        
        Args:
            image: Input RGB image
            
        Returns:
            Tuple of (processed_image, edge_case_info)
        """
        edge_cases = self.detect_edge_cases(image)
        result = image.copy()
        
        # Handle each detected edge case
        if edge_cases['is_blown_out']:
            result = self.handle_blown_out(result)
            logger.warning("Handling blown out image")
        
        if edge_cases['is_too_dark']:
            result = self.handle_extremely_dark(result)
            logger.warning("Handling extremely dark image")
        
        if edge_cases['has_motion_blur']:
            result = self.handle_motion_blur(result)
            logger.warning("Handling motion blur")
        
        if edge_cases['has_mixed_lighting']:
            result = self.handle_mixed_lighting(result)
            logger.warning("Handling mixed lighting")
        
        if edge_cases['has_flash_artifacts']:
            result = self.handle_flash_artifacts(result)
            logger.warning("Handling flash artifacts")
        
        if edge_cases['has_thermal_noise']:
            result = self.handle_thermal_noise(result)
            logger.warning("Handling thermal noise")
        
        return result, edge_cases
    
    def detect_edge_cases(self, image: np.ndarray) -> Dict:
        """
        Detect various edge cases in the image
        
        Args:
            image: Input image
            
        Returns:
            Dictionary of detected edge cases
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        edge_cases = {
            'is_blown_out': self.detect_blown_out(gray),
            'is_too_dark': self.detect_extremely_dark(gray),
            'has_motion_blur': self.detect_motion_blur(gray),
            'has_mixed_lighting': self.detect_mixed_lighting(image),
            'has_flash_artifacts': self.detect_flash_artifacts(image),
            'has_thermal_noise': self.detect_thermal_noise(image),
            'severity': 'none'
        }
        
        # Determine overall severity
        num_issues = sum([v for k, v in edge_cases.items() 
                         if k != 'severity' and isinstance(v, bool)])
        
        if num_issues >= 3:
            edge_cases['severity'] = 'severe'
        elif num_issues >= 2:
            edge_cases['severity'] = 'moderate'
        elif num_issues >= 1:
            edge_cases['severity'] = 'mild'
        
        return edge_cases
    
    def detect_blown_out(self, gray: np.ndarray) -> bool:
        """
        Detect if image is mostly blown out (overexposed)
        
        Args:
            gray: Grayscale image
            
        Returns:
            True if image is blown out
        """
        # Check percentage of saturated pixels
        saturated = np.sum(gray > 250) / gray.size
        
        # Check if histogram is heavily skewed to bright
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hist_norm = hist / hist.sum()
        bright_ratio = np.sum(hist_norm[200:])
        
        return saturated > 0.3 or bright_ratio > 0.5
    
    def detect_extremely_dark(self, gray: np.ndarray) -> bool:
        """
        Detect if image is extremely dark
        
        Args:
            gray: Grayscale image
            
        Returns:
            True if image is too dark
        """
        # Check mean brightness
        mean_brightness = np.mean(gray)
        
        # Check percentage of dark pixels
        dark_pixels = np.sum(gray < 30) / gray.size
        
        return mean_brightness < 30 or dark_pixels > 0.7
    
    def detect_motion_blur(self, gray: np.ndarray) -> bool:
        """
        Detect motion blur using Laplacian variance
        
        Args:
            gray: Grayscale image
            
        Returns:
            True if motion blur detected
        """
        # Calculate Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        # Low variance indicates blur
        return variance < self.max_blur_score
    
    def detect_mixed_lighting(self, image: np.ndarray) -> bool:
        """
        Detect mixed lighting conditions (e.g., indoor/outdoor mix)
        
        Args:
            image: RGB image
            
        Returns:
            True if mixed lighting detected
        """
        # Divide image into quadrants
        h, w = image.shape[:2]
        quadrants = [
            image[:h//2, :w//2],
            image[:h//2, w//2:],
            image[h//2:, :w//2],
            image[h//2:, w//2:]
        ]
        
        # Analyze color temperature of each quadrant
        temperatures = []
        for quad in quadrants:
            # Simple color temperature estimation
            mean_rgb = np.mean(quad.reshape(-1, 3), axis=0)
            # Ratio of blue to red indicates temperature
            if mean_rgb[0] > 0:
                temp_ratio = mean_rgb[2] / mean_rgb[0]
            else:
                temp_ratio = 1.0
            temperatures.append(temp_ratio)
        
        # High variance in color temperature indicates mixed lighting
        temp_variance = np.std(temperatures)
        
        return temp_variance > 0.3
    
    def detect_flash_artifacts(self, image: np.ndarray) -> bool:
        """
        Detect flash photography artifacts
        
        Args:
            image: RGB image
            
        Returns:
            True if flash artifacts detected
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        
        # Check for bright center (typical flash pattern)
        center_region = gray[h//3:2*h//3, w//3:2*w//3]
        edge_region = np.concatenate([
            gray[:h//3, :].flatten(),
            gray[2*h//3:, :].flatten(),
            gray[:, :w//3].flatten(),
            gray[:, 2*w//3:].flatten()
        ])
        
        center_mean = np.mean(center_region)
        edge_mean = np.mean(edge_region)
        
        # Flash typically makes center much brighter
        brightness_ratio = center_mean / (edge_mean + 1e-6)
        
        # Check for harsh shadows (high local contrast)
        local_contrast = np.std(gray)
        
        return brightness_ratio > 1.5 and local_contrast > 60
    
    def detect_thermal_noise(self, image: np.ndarray) -> bool:
        """
        Detect thermal noise from overheated phone sensor
        
        Args:
            image: RGB image
            
        Returns:
            True if thermal noise detected
        """
        # Thermal noise appears as random hot pixels
        # Check for salt-and-pepper noise pattern
        
        # Calculate local variance
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Use small kernel to detect pixel-level noise
        kernel_size = 3
        mean = cv2.blur(gray, (kernel_size, kernel_size))
        sq_mean = cv2.blur(gray**2, (kernel_size, kernel_size))
        variance = sq_mean - mean**2
        
        # High variance at pixel level indicates noise
        noise_pixels = np.sum(variance > 100) / variance.size
        
        return noise_pixels > 0.05
    
    def handle_blown_out(self, image: np.ndarray) -> np.ndarray:
        """
        Handle severely overexposed images
        
        Args:
            image: Blown out image
            
        Returns:
            Recovered image
        """
        # Aggressive highlight recovery
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0].astype(np.float32)
        
        # Compress highlights aggressively
        high_values = l_channel > 200
        if np.any(high_values):
            l_channel[high_values] = 200 + (l_channel[high_values] - 200) * 0.2
        
        # Apply strong gamma correction
        l_channel = self.apply_gamma_channel(l_channel / 255.0, 0.5) * 255
        
        # Enhance remaining detail
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(np.clip(l_channel, 0, 255).astype(np.uint8))
        
        lab[:, :, 0] = l_enhanced
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return result
    
    def handle_extremely_dark(self, image: np.ndarray) -> np.ndarray:
        """
        Handle extremely dark images
        
        Args:
            image: Very dark image
            
        Returns:
            Brightened image
        """
        # Adaptive histogram equalization with strong parameters
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]
        
        # Strong CLAHE
        clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(4, 4))
        l_enhanced = clahe.apply(l_channel)
        
        # Additional gamma correction
        l_enhanced = self.apply_gamma_channel(l_enhanced / 255.0, 2.0) * 255
        
        lab[:, :, 0] = np.clip(l_enhanced, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Denoise (dark images are often noisy)
        result = cv2.fastNlMeansDenoisingColored(result, None, 10, 10, 7, 21)
        
        return result
    
    def handle_motion_blur(self, image: np.ndarray) -> np.ndarray:
        """
        Handle motion blur (limited recovery possible)
        
        Args:
            image: Blurred image
            
        Returns:
            Sharpened image
        """
        # Unsharp masking
        gaussian = cv2.GaussianBlur(image, (5, 5), 1.0)
        sharpened = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
        
        # Edge enhancement
        kernel = np.array([[-1, -1, -1],
                          [-1, 9, -1],
                          [-1, -1, -1]])
        enhanced = cv2.filter2D(sharpened, -1, kernel)
        
        # Blend to avoid over-sharpening
        result = cv2.addWeighted(sharpened, 0.7, enhanced, 0.3, 0)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def handle_mixed_lighting(self, image: np.ndarray) -> np.ndarray:
        """
        Handle mixed lighting conditions
        
        Args:
            image: Image with mixed lighting
            
        Returns:
            Balanced image
        """
        # Use local adaptive processing
        from .local_adaptive import LocalAdaptiveProcessor
        
        processor = LocalAdaptiveProcessor(grid_size=(3, 3))
        result = processor.process(image)
        
        # Additional color balancing
        lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB)
        
        # Reduce color casts in A and B channels
        lab[:, :, 1] = np.clip(lab[:, :, 1] * 0.8 + 128 * 0.2, 0, 255).astype(np.uint8)
        lab[:, :, 2] = np.clip(lab[:, :, 2] * 0.8 + 128 * 0.2, 0, 255).astype(np.uint8)
        
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return result
    
    def handle_flash_artifacts(self, image: np.ndarray) -> np.ndarray:
        """
        Handle flash photography artifacts
        
        Args:
            image: Image with flash artifacts
            
        Returns:
            Corrected image
        """
        h, w = image.shape[:2]
        
        # Create vignetting correction mask (inverse of flash pattern)
        center = (w // 2, h // 2)
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
        max_dist = np.sqrt(center[0]**2 + center[1]**2)
        
        # Create correction mask
        vignette_mask = 0.7 + 0.3 * (dist / max_dist)
        
        # Apply correction
        result = image.astype(np.float32)
        for c in range(3):
            result[:, :, c] *= vignette_mask
        
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        # Reduce harsh shadows
        from .shadow_highlight import ShadowHighlightRecovery
        sh_processor = ShadowHighlightRecovery()
        result = sh_processor.process(result)
        
        return result
    
    def handle_thermal_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Handle thermal noise from overheated sensor
        
        Args:
            image: Noisy image
            
        Returns:
            Denoised image
        """
        # Non-local means denoising (good for thermal noise)
        denoised = cv2.fastNlMeansDenoisingColored(
            image, None,
            h=15,  # Filter strength
            hColor=15,  # Filter strength for color
            templateWindowSize=7,
            searchWindowSize=21
        )
        
        # Additional bilateral filtering for edge preservation
        result = cv2.bilateralFilter(denoised, 9, 75, 75)
        
        return result
    
    def apply_gamma_channel(self, channel: np.ndarray, gamma: float) -> np.ndarray:
        """
        Apply gamma correction to normalized channel
        
        Args:
            channel: Normalized channel (0-1)
            gamma: Gamma value
            
        Returns:
            Gamma corrected channel (0-1)
        """
        return np.power(channel, 1.0 / gamma)
    
    def validate_output(self, image: np.ndarray) -> bool:
        """
        Validate that output image is usable
        
        Args:
            image: Processed image
            
        Returns:
            True if image is usable
        """
        # Check if image has sufficient detail
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Check dynamic range
        min_val, max_val = np.min(gray), np.max(gray)
        dynamic_range = max_val - min_val
        
        # Check variance (indicates detail)
        variance = np.var(gray)
        
        # Image is usable if it has reasonable dynamic range and detail
        return dynamic_range > 50 and variance > 100