"""
Retinex-based Illumination Normalization for Plant Disease Detection
Addresses the critical 45-68% field accuracy drop due to lighting variations
Target: Reduce lighting variance from 35% to <10% while preserving disease patterns
"""

import numpy as np
import cv2
from typing import Tuple, Dict, Optional, List
import time
import logging

logger = logging.getLogger(__name__)


class RetinexIllumination:
    """
    Multi-scale Retinex with disease pattern preservation
    Separates illumination (L) and reflectance (R) where Image = L × R
    
    Key features:
    - LAB color space processing (preserve disease colors in A/B channels)
    - Adaptive parameters based on lighting conditions
    - Disease pattern enhancement
    - 150-200ms processing time target
    """
    
    def __init__(self, 
                 clahe_clip_limit: float = 3.0,
                 clahe_tile_size: Tuple[int, int] = (8, 8)):
        """
        Initialize Retinex processor
        
        Args:
            clahe_clip_limit: Contrast limiting for CLAHE (default 3.0 for disease preservation)
            clahe_tile_size: Tile grid size for CLAHE
        """
        # Multi-scale parameters (small, medium, large)
        self.scales = [15, 80, 250]
        self.weights = [1/3, 1/3, 1/3]
        
        # CLAHE for local contrast enhancement
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=clahe_tile_size
        )
        
        # Adaptive parameters cache
        self.adaptive_params = {}
        
        # Performance stats
        self.stats = {
            'processed_count': 0,
            'avg_time_ms': 0,
            'variance_reduction': []
        }
    
    def process(self, 
                image: np.ndarray,
                preserve_disease: bool = True,
                use_multi_scale: bool = True) -> np.ndarray:
        """
        Apply Retinex illumination normalization
        
        Args:
            image: Input image (RGB, uint8)
            preserve_disease: Whether to enhance disease patterns
            use_multi_scale: Use MSR (True) or SSR (False)
        
        Returns:
            Normalized image with reduced lighting variance
        """
        start_time = time.time()
        
        # Analyze lighting condition for adaptive parameters
        lighting_condition, params = self.analyze_lighting_condition(image)
        logger.info(f"Detected lighting: {lighting_condition}")
        
        # Convert to LAB for illumination processing
        lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Separate channels
        l_channel = lab_image[:, :, 0].astype(np.float32)
        a_channel = lab_image[:, :, 1]
        b_channel = lab_image[:, :, 2]
        
        # Apply Retinex to L channel only (preserve disease colors in A/B)
        if use_multi_scale:
            l_normalized = self.multi_scale_retinex(l_channel, params)
        else:
            l_normalized = self.single_scale_retinex(l_channel, sigma=80)
        
        # Apply CLAHE for local contrast
        l_normalized_uint8 = np.clip(l_normalized, 0, 255).astype(np.uint8)
        l_enhanced = self.clahe.apply(l_normalized_uint8)
        
        # Apply adaptive gamma correction based on lighting
        l_corrected = self.apply_adaptive_gamma(l_enhanced, params['gamma'])
        
        # Reconstruct LAB image
        lab_normalized = np.stack([l_corrected, a_channel, b_channel], axis=2)
        
        # Convert back to RGB
        result = cv2.cvtColor(lab_normalized, cv2.COLOR_LAB2RGB)
        
        # Preserve/enhance disease patterns if requested
        if preserve_disease:
            result = self.preserve_disease_patterns(image, result)
        
        # Update statistics
        elapsed_ms = (time.time() - start_time) * 1000
        self._update_stats(elapsed_ms, image, result)
        
        logger.info(f"Retinex processing completed in {elapsed_ms:.1f}ms")
        
        return result
    
    def single_scale_retinex(self, 
                            channel: np.ndarray, 
                            sigma: float = 80) -> np.ndarray:
        """
        Single-Scale Retinex (SSR) implementation
        
        Args:
            channel: Single channel image (float32)
            sigma: Gaussian blur sigma for illumination estimation
        
        Returns:
            Normalized channel with illumination removed
        """
        # Add small constant to avoid log(0)
        img_log = np.log(channel + 1.0)
        
        # Estimate illumination using Gaussian blur
        # Larger sigma captures global illumination
        illumination = cv2.GaussianBlur(channel, (0, 0), sigma)
        illumination_log = np.log(illumination + 1.0)
        
        # Separate reflectance (detail) from illumination
        reflectance_log = img_log - illumination_log
        
        # Convert back from log domain
        reflectance = np.exp(reflectance_log)
        
        # Normalize to [0, 255] range
        return self.normalize_channel(reflectance)
    
    def multi_scale_retinex(self, 
                           channel: np.ndarray,
                           params: Dict) -> np.ndarray:
        """
        Multi-Scale Retinex (MSR) for better detail preservation
        
        Args:
            channel: Single channel image (float32)
            params: Adaptive parameters based on lighting
        
        Returns:
            Multi-scale normalized channel
        """
        msr_result = np.zeros_like(channel, dtype=np.float32)
        
        # Apply SSR at multiple scales
        for sigma, weight in zip(self.scales, self.weights):
            # Adjust sigma based on lighting condition
            adjusted_sigma = sigma * params.get('sigma_mult', 1.0)
            
            # Apply single-scale Retinex
            ssr = self.single_scale_retinex(channel, adjusted_sigma)
            
            # Weighted combination
            msr_result += weight * ssr
        
        return msr_result
    
    def analyze_lighting_condition(self, image: np.ndarray) -> Tuple[str, Dict]:
        """
        Analyze image histogram to determine lighting condition
        
        Args:
            image: Input RGB image
        
        Returns:
            Tuple of (condition_name, adaptive_parameters)
        """
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Calculate histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hist_norm = hist / hist.sum()
        
        # Calculate statistics
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        
        # Determine percentiles
        cumsum = np.cumsum(hist_norm)
        p5 = np.searchsorted(cumsum, 0.05)
        p95 = np.searchsorted(cumsum, 0.95)
        
        # Classify lighting condition
        if mean_brightness > 180 and p95 > 240:
            # Overexposed (harsh sunlight)
            condition = 'overexposed'
            params = {
                'gamma': 0.7,
                'clahe_clip': 2.0,
                'sigma_mult': 1.2,
                'shadow_boost': 1.5
            }
        elif mean_brightness < 60 and p95 < 150:
            # Underexposed (deep shade)
            condition = 'underexposed'
            params = {
                'gamma': 1.5,
                'clahe_clip': 4.0,
                'sigma_mult': 0.8,
                'shadow_boost': 2.0
            }
        elif std_brightness > 70:
            # High contrast (mixed lighting)
            condition = 'high_contrast'
            params = {
                'gamma': 1.0,
                'clahe_clip': 3.5,
                'sigma_mult': 1.0,
                'shadow_boost': 1.3
            }
        else:
            # Normal lighting
            condition = 'normal'
            params = {
                'gamma': 1.0,
                'clahe_clip': 3.0,
                'sigma_mult': 1.0,
                'shadow_boost': 1.0
            }
        
        logger.debug(f"Lighting: {condition}, Mean: {mean_brightness:.1f}, Std: {std_brightness:.1f}")
        
        return condition, params
    
    def apply_adaptive_gamma(self, 
                            channel: np.ndarray, 
                            gamma: float) -> np.ndarray:
        """
        Apply adaptive gamma correction
        
        Args:
            channel: Single channel image (uint8)
            gamma: Gamma value (< 1 darkens, > 1 brightens)
        
        Returns:
            Gamma corrected channel
        """
        # Build lookup table
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in range(256)]).astype(np.uint8)
        
        # Apply lookup table
        return cv2.LUT(channel, table)
    
    def preserve_disease_patterns(self, 
                                 original: np.ndarray, 
                                 processed: np.ndarray) -> np.ndarray:
        """
        Ensure disease patterns are preserved/enhanced after normalization
        
        Args:
            original: Original image with disease patterns
            processed: Retinex-processed image
        
        Returns:
            Image with enhanced disease patterns
        """
        # Detect disease regions in original
        disease_mask = self.detect_disease_regions(original)
        
        # Enhance local contrast in disease regions
        disease_enhanced = self.enhance_disease_contrast(processed, disease_mask)
        
        # Blend: stronger enhancement in disease areas
        alpha = disease_mask[:, :, np.newaxis].astype(np.float32) / 255.0
        result = (1 - alpha) * processed + alpha * disease_enhanced
        
        return result.astype(np.uint8)
    
    def detect_disease_regions(self, image: np.ndarray) -> np.ndarray:
        """
        Detect potential disease regions for preservation
        
        Args:
            image: Input RGB image
        
        Returns:
            Binary mask of disease regions (0-255)
        """
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        masks = []
        
        # Brown/dark spots (blight, necrosis)
        lower_brown = np.array([10, 50, 20])
        upper_brown = np.array([20, 255, 150])
        brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
        masks.append(brown_mask)
        
        # Yellow areas (chlorosis, mosaic virus)
        lower_yellow = np.array([20, 50, 100])
        upper_yellow = np.array([30, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        masks.append(yellow_mask)
        
        # White/gray areas (powdery mildew)
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        masks.append(white_mask)
        
        # Dark spots (various diseases)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, dark_mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
        masks.append(dark_mask)
        
        # Combine all masks
        combined = np.zeros_like(masks[0], dtype=np.uint8)
        for mask in masks:
            combined = cv2.bitwise_or(combined, mask)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Dilate slightly to include boundaries
        combined = cv2.dilate(combined, kernel, iterations=1)
        
        return combined
    
    def enhance_disease_contrast(self, 
                                image: np.ndarray, 
                                mask: np.ndarray) -> np.ndarray:
        """
        Enhance contrast specifically in disease regions
        
        Args:
            image: RGB image
            mask: Binary mask of disease regions
        
        Returns:
            Image with enhanced disease regions
        """
        result = image.copy()
        
        # Apply stronger CLAHE in disease regions
        lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB)
        
        # Create stronger CLAHE for disease areas
        strong_clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4, 4))
        
        # Apply to masked regions
        l_channel = lab[:, :, 0]
        l_enhanced = strong_clahe.apply(l_channel)
        
        # Blend enhanced version in disease areas only
        lab[:, :, 0] = np.where(mask > 0, l_enhanced, l_channel)
        
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return result
    
    def normalize_channel(self, channel: np.ndarray) -> np.ndarray:
        """
        Normalize channel to [0, 255] range preserving relative values
        
        Args:
            channel: Float channel data
        
        Returns:
            Normalized channel as float32 in [0, 255]
        """
        # Remove outliers (top/bottom 1%)
        p1, p99 = np.percentile(channel, [1, 99])
        channel_clipped = np.clip(channel, p1, p99)
        
        # Normalize to [0, 255]
        min_val = channel_clipped.min()
        max_val = channel_clipped.max()
        
        if max_val - min_val > 0:
            normalized = ((channel_clipped - min_val) / (max_val - min_val)) * 255
        else:
            normalized = channel_clipped
        
        return normalized.astype(np.float32)
    
    def calculate_variance(self, image: np.ndarray) -> float:
        """
        Calculate lighting variance as a percentage
        
        Args:
            image: Input image
        
        Returns:
            Variance percentage
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        mean = np.mean(gray)
        std = np.std(gray)
        
        # Coefficient of variation as percentage
        if mean > 0:
            cv = (std / mean) * 100
        else:
            cv = 0
        
        return cv
    
    def _update_stats(self, 
                     time_ms: float, 
                     original: np.ndarray, 
                     processed: np.ndarray):
        """
        Update processing statistics
        """
        self.stats['processed_count'] += 1
        n = self.stats['processed_count']
        
        # Update average time
        prev_avg = self.stats['avg_time_ms']
        self.stats['avg_time_ms'] = (prev_avg * (n - 1) + time_ms) / n
        
        # Calculate variance reduction
        var_before = self.calculate_variance(original)
        var_after = self.calculate_variance(processed)
        reduction = ((var_before - var_after) / var_before) * 100 if var_before > 0 else 0
        
        self.stats['variance_reduction'].append(reduction)
        
        logger.debug(f"Variance: {var_before:.1f}% -> {var_after:.1f}% (reduction: {reduction:.1f}%)")
    
    def get_stats(self) -> Dict:
        """Get processing statistics"""
        stats = self.stats.copy()
        if self.stats['variance_reduction']:
            stats['avg_variance_reduction'] = np.mean(self.stats['variance_reduction'])
        return stats