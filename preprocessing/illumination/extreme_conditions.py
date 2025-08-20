"""
Extreme Lighting Handler for Field Conditions
Addresses the 79% sun vs 89% indoor accuracy gap by handling harsh lighting
Target: Reduce accuracy gap from 10% to <5%
"""

import numpy as np
import cv2
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class ExtremeLightingHandler:
    """
    Handles extreme lighting conditions common in field photography
    Specifically targets harsh direct sunlight which causes:
    - Specular highlights on leaves (blown out regions)
    - Deep shadows simultaneously
    - Color cast from intense sun
    - Loss of disease pattern detail
    """
    
    def __init__(self):
        """Initialize extreme lighting handler"""
        # HDR tone mapping parameters
        self.tone_mapper = cv2.createTonemapReinhard(gamma=1.0, intensity=0.0, 
                                                     light_adapt=0.8, color_adapt=0.0)
        
        # Specular detection thresholds
        self.specular_threshold = 240  # Pixels above this are likely specular
        self.shadow_threshold = 30     # Pixels below this are deep shadows
        
        # Recovery parameters
        self.highlight_recovery_strength = 0.7
        self.shadow_recovery_strength = 0.5
        
    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Process image with extreme lighting handling
        
        Args:
            image: Input RGB image with potential extreme lighting
            
        Returns:
            Processed image with recovered highlights and shadows
        """
        # Analyze lighting severity
        severity = self.analyze_lighting_severity(image)
        
        if severity['is_extreme']:
            logger.info(f"Extreme lighting detected: {severity['type']}")
            
            # Apply appropriate recovery based on severity
            if severity['has_specular']:
                image = self.suppress_specular_highlights(image)
            
            if severity['has_deep_shadows']:
                image = self.recover_shadows(image)
            
            if severity['needs_hdr']:
                image = self.apply_hdr_tone_mapping(image)
            
            # Color cast correction for harsh sun
            if severity['has_color_cast']:
                image = self.correct_sun_color_cast(image)
        
        return image
    
    def analyze_lighting_severity(self, image: np.ndarray) -> Dict:
        """
        Analyze how extreme the lighting conditions are
        
        Args:
            image: Input RGB image
            
        Returns:
            Dictionary with severity analysis
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Calculate histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hist_norm = hist / hist.sum()
        
        # Check for blown highlights (specular)
        specular_ratio = np.sum(hist_norm[self.specular_threshold:])
        has_specular = specular_ratio > 0.05  # >5% pixels are blown
        
        # Check for deep shadows
        shadow_ratio = np.sum(hist_norm[:self.shadow_threshold])
        has_deep_shadows = shadow_ratio > 0.15  # >15% pixels are very dark
        
        # Check for simultaneous highlights and shadows (harsh sun signature)
        has_both = has_specular and has_deep_shadows
        
        # Calculate dynamic range
        cumsum = np.cumsum(hist_norm)
        p5 = np.searchsorted(cumsum, 0.05)
        p95 = np.searchsorted(cumsum, 0.95)
        dynamic_range = p95 - p5
        
        # Check for color cast (common in golden hour/harsh sun)
        mean_rgb = np.mean(image.reshape(-1, 3), axis=0)
        color_variance = np.std(mean_rgb)
        has_color_cast = color_variance > 30  # Significant channel imbalance
        
        severity = {
            'is_extreme': has_both or (dynamic_range > 200),
            'has_specular': has_specular,
            'has_deep_shadows': has_deep_shadows,
            'needs_hdr': has_both,
            'has_color_cast': has_color_cast,
            'type': 'harsh_sun' if has_both else ('overexposed' if has_specular else 'underexposed'),
            'specular_ratio': specular_ratio,
            'shadow_ratio': shadow_ratio,
            'dynamic_range': dynamic_range
        }
        
        return severity
    
    def suppress_specular_highlights(self, image: np.ndarray) -> np.ndarray:
        """
        Suppress specular highlights while preserving disease patterns
        Critical for maintaining powdery mildew visibility
        
        Args:
            image: Input RGB image with specular highlights
            
        Returns:
            Image with suppressed speculars
        """
        # Convert to float for processing
        img_float = image.astype(np.float32) / 255.0
        
        # Detect specular regions
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        specular_mask = (gray > self.specular_threshold).astype(np.float32)
        
        # Dilate mask slightly to include edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        specular_mask = cv2.dilate(specular_mask, kernel, iterations=1)
        
        # Gaussian blur for smooth transition
        specular_mask = cv2.GaussianBlur(specular_mask, (15, 15), 0)
        
        # Inpainting-inspired approach: use surrounding pixels
        for c in range(3):
            channel = img_float[:, :, c]
            
            # Get non-specular mean for this channel
            non_specular_pixels = channel[specular_mask < 0.5]
            if len(non_specular_pixels) > 0:
                replacement_value = np.percentile(non_specular_pixels, 90)
            else:
                replacement_value = 0.8
            
            # Blend original with replacement based on mask
            channel_recovered = channel * (1 - specular_mask * self.highlight_recovery_strength) + \
                              replacement_value * specular_mask * self.highlight_recovery_strength
            
            img_float[:, :, c] = channel_recovered
        
        # Convert back to uint8
        result = np.clip(img_float * 255, 0, 255).astype(np.uint8)
        
        return result
    
    def recover_shadows(self, image: np.ndarray) -> np.ndarray:
        """
        Recover detail in shadow regions without affecting disease spots
        Preserves dark disease patterns (blight, necrosis)
        
        Args:
            image: Input RGB image with deep shadows
            
        Returns:
            Image with recovered shadow detail
        """
        # Work in LAB space to preserve color
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0].astype(np.float32)
        
        # Detect shadow regions
        shadow_mask = (l_channel < self.shadow_threshold).astype(np.float32)
        
        # Smooth the mask
        shadow_mask = cv2.GaussianBlur(shadow_mask, (21, 21), 0)
        
        # Adaptive shadow lifting using log curve
        # This preserves relative differences (important for disease patterns)
        shadow_boost = np.log1p(l_channel / 255.0) * 255.0
        
        # Blend based on shadow mask
        l_recovered = l_channel * (1 - shadow_mask * self.shadow_recovery_strength) + \
                     shadow_boost * shadow_mask * self.shadow_recovery_strength
        
        # Ensure we don't over-brighten
        l_recovered = np.clip(l_recovered, 0, 255)
        
        # Apply local contrast enhancement in shadow regions
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        l_recovered_uint8 = l_recovered.astype(np.uint8)
        l_enhanced = clahe.apply(l_recovered_uint8)
        
        # Blend enhanced version only in shadow areas
        l_final = np.where(shadow_mask > 0.3, l_enhanced, l_recovered_uint8)
        
        # Reconstruct image
        lab[:, :, 0] = l_final
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return result
    
    def apply_hdr_tone_mapping(self, image: np.ndarray) -> np.ndarray:
        """
        Apply HDR-like tone mapping for extreme dynamic range
        Compresses range while preserving local contrast
        
        Args:
            image: Input RGB image
            
        Returns:
            Tone-mapped image
        """
        # Convert to float32 for tone mapping
        img_float = image.astype(np.float32) / 255.0
        
        # Apply Reinhard tone mapping
        tone_mapped = self.tone_mapper.process(img_float)
        
        # Enhance local contrast after tone mapping
        # This helps maintain disease pattern visibility
        lab = cv2.cvtColor(tone_mapped, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]
        
        # Adaptive histogram equalization on L channel
        l_uint8 = (l_channel * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_uint8)
        
        lab[:, :, 0] = l_enhanced / 255.0
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Convert back to uint8
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        
        return result
    
    def correct_sun_color_cast(self, image: np.ndarray) -> np.ndarray:
        """
        Correct color cast from harsh sun (golden/orange tint)
        Preserves disease-specific colors
        
        Args:
            image: Input RGB image with color cast
            
        Returns:
            Color-corrected image
        """
        # Estimate illuminant color using gray world assumption
        # But protect disease-specific colors
        
        # Create mask for likely neutral areas (not disease)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Mask out strong colors (likely disease or vegetation)
        saturation = hsv[:, :, 1]
        neutral_mask = saturation < 50  # Low saturation = likely neutral
        
        if np.sum(neutral_mask) > 100:  # Enough neutral pixels
            # Estimate illuminant from neutral areas
            neutral_pixels = image[neutral_mask]
            mean_rgb = np.mean(neutral_pixels, axis=0)
            
            # Calculate correction factors
            gray_value = np.mean(mean_rgb)
            correction = gray_value / (mean_rgb + 1e-6)
            
            # Limit correction strength to avoid overcorrection
            correction = np.clip(correction, 0.7, 1.3)
            
            # Apply correction
            corrected = image.astype(np.float32)
            for c in range(3):
                corrected[:, :, c] *= correction[c]
            
            result = np.clip(corrected, 0, 255).astype(np.uint8)
        else:
            # Not enough neutral pixels, use simple white balance
            result = self.simple_white_balance(image)
        
        return result
    
    def simple_white_balance(self, image: np.ndarray) -> np.ndarray:
        """
        Simple percentile-based white balance
        
        Args:
            image: Input RGB image
            
        Returns:
            White-balanced image
        """
        result = np.zeros_like(image)
        
        for c in range(3):
            channel = image[:, :, c]
            
            # Use 5th and 95th percentiles
            p5, p95 = np.percentile(channel, [5, 95])
            
            # Scale to full range
            channel_scaled = np.interp(channel, [p5, p95], [0, 255])
            result[:, :, c] = np.clip(channel_scaled, 0, 255)
        
        return result.astype(np.uint8)
    
    def detect_iphone_auto_exposure(self, image: np.ndarray, 
                                    metadata: Optional[Dict] = None) -> bool:
        """
        Detect if iPhone auto-exposure has created problems
        
        Args:
            image: Input image
            metadata: Optional EXIF metadata
            
        Returns:
            True if auto-exposure issues detected
        """
        # Check for telltale signs of aggressive auto-exposure
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Histogram should not be too centered (over-corrected)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hist_norm = hist / hist.sum()
        
        # Calculate histogram entropy
        entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))
        
        # Low entropy suggests aggressive auto-exposure
        return entropy < 6.5