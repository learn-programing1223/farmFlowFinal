"""
Shadow and Highlight Recovery for Disease Pattern Preservation
Recovers detail in extreme regions without affecting disease visibility
"""

import numpy as np
import cv2
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ShadowHighlightRecovery:
    """
    Advanced shadow lifting and highlight compression
    Preserves disease patterns while recovering detail in extreme regions
    """
    
    def __init__(self):
        """Initialize shadow/highlight recovery"""
        self.shadow_threshold = 50
        self.highlight_threshold = 205
        
        # Adaptive parameters
        self.max_shadow_boost = 2.5
        self.max_highlight_compression = 0.6
        
    def process(self, image: np.ndarray, 
                preserve_disease: bool = True) -> np.ndarray:
        """
        Apply shadow and highlight recovery
        
        Args:
            image: Input RGB image
            preserve_disease: Whether to preserve disease patterns
            
        Returns:
            Image with recovered shadows and highlights
        """
        # Analyze image to determine recovery needs
        needs_shadow, needs_highlight = self.analyze_recovery_needs(image)
        
        result = image.copy()
        
        if needs_shadow:
            result = self.recover_shadows(result, preserve_disease)
            logger.debug("Applied shadow recovery")
        
        if needs_highlight:
            result = self.recover_highlights(result, preserve_disease)
            logger.debug("Applied highlight recovery")
        
        return result
    
    def analyze_recovery_needs(self, image: np.ndarray) -> Tuple[bool, bool]:
        """
        Determine if shadow/highlight recovery is needed
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (needs_shadow_recovery, needs_highlight_recovery)
        """
        # Convert to luminance
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Calculate histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hist_norm = hist / hist.sum()
        
        # Check shadow regions
        shadow_ratio = np.sum(hist_norm[:self.shadow_threshold])
        needs_shadow = shadow_ratio > 0.1  # >10% in shadows
        
        # Check highlight regions
        highlight_ratio = np.sum(hist_norm[self.highlight_threshold:])
        needs_highlight = highlight_ratio > 0.1  # >10% in highlights
        
        return needs_shadow, needs_highlight
    
    def recover_shadows(self, image: np.ndarray, 
                       preserve_disease: bool = True) -> np.ndarray:
        """
        Recover shadow detail while preserving dark disease patterns
        
        Args:
            image: Input RGB image
            preserve_disease: Whether to preserve disease patterns
            
        Returns:
            Image with recovered shadow detail
        """
        # Work in LAB space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0].astype(np.float32)
        
        # Create shadow mask
        shadow_mask = self.create_shadow_mask(l_channel)
        
        if preserve_disease:
            # Detect disease regions to protect
            disease_mask = self.detect_dark_disease_patterns(image)
            # Reduce shadow recovery in disease areas
            shadow_mask = shadow_mask * (1 - disease_mask * 0.7)
        
        # Apply adaptive shadow lifting
        l_recovered = self.adaptive_shadow_lift(l_channel, shadow_mask)
        
        # Enhance local contrast in lifted regions
        l_enhanced = self.enhance_shadow_contrast(l_recovered, shadow_mask)
        
        # Update LAB image
        lab[:, :, 0] = np.clip(l_enhanced, 0, 255).astype(np.uint8)
        
        # Convert back to RGB
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return result
    
    def recover_highlights(self, image: np.ndarray,
                          preserve_disease: bool = True) -> np.ndarray:
        """
        Recover highlight detail while preserving bright disease patterns
        
        Args:
            image: Input RGB image
            preserve_disease: Whether to preserve disease patterns
            
        Returns:
            Image with recovered highlight detail
        """
        # Work in LAB space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0].astype(np.float32)
        
        # Create highlight mask
        highlight_mask = self.create_highlight_mask(l_channel)
        
        if preserve_disease:
            # Detect bright disease patterns (e.g., powdery mildew)
            disease_mask = self.detect_bright_disease_patterns(image)
            # Reduce highlight compression in disease areas
            highlight_mask = highlight_mask * (1 - disease_mask * 0.7)
        
        # Apply highlight compression
        l_compressed = self.compress_highlights(l_channel, highlight_mask)
        
        # Recover color in highlight regions
        result_lab = lab.copy()
        result_lab[:, :, 0] = np.clip(l_compressed, 0, 255).astype(np.uint8)
        
        # Enhance A/B channels in highlight regions for color recovery
        if np.any(highlight_mask > 0.5):
            result_lab = self.recover_highlight_color(result_lab, highlight_mask)
        
        # Convert back to RGB
        result = cv2.cvtColor(result_lab, cv2.COLOR_LAB2RGB)
        
        return result
    
    def create_shadow_mask(self, l_channel: np.ndarray) -> np.ndarray:
        """
        Create mask for shadow regions
        
        Args:
            l_channel: Luminance channel
            
        Returns:
            Float mask (0-1) indicating shadow strength
        """
        # Normalize to 0-1
        l_norm = l_channel / 255.0
        
        # Create smooth shadow mask
        shadow_mask = np.zeros_like(l_norm)
        
        # Strong shadows (very dark)
        very_dark = l_norm < 0.1
        shadow_mask[very_dark] = 1.0
        
        # Medium shadows
        medium_dark = (l_norm >= 0.1) & (l_norm < 0.2)
        shadow_mask[medium_dark] = 1.0 - (l_norm[medium_dark] - 0.1) * 10
        
        # Smooth the mask
        shadow_mask = cv2.GaussianBlur(shadow_mask, (21, 21), 0)
        
        return shadow_mask
    
    def create_highlight_mask(self, l_channel: np.ndarray) -> np.ndarray:
        """
        Create mask for highlight regions
        
        Args:
            l_channel: Luminance channel
            
        Returns:
            Float mask (0-1) indicating highlight strength
        """
        # Normalize to 0-1
        l_norm = l_channel / 255.0
        
        # Create smooth highlight mask
        highlight_mask = np.zeros_like(l_norm)
        
        # Strong highlights (very bright)
        very_bright = l_norm > 0.9
        highlight_mask[very_bright] = 1.0
        
        # Medium highlights
        medium_bright = (l_norm >= 0.8) & (l_norm <= 0.9)
        highlight_mask[medium_bright] = (l_norm[medium_bright] - 0.8) * 10
        
        # Smooth the mask
        highlight_mask = cv2.GaussianBlur(highlight_mask, (21, 21), 0)
        
        return highlight_mask
    
    def adaptive_shadow_lift(self, l_channel: np.ndarray,
                            shadow_mask: np.ndarray) -> np.ndarray:
        """
        Apply adaptive shadow lifting based on local characteristics
        
        Args:
            l_channel: Luminance channel
            shadow_mask: Shadow mask
            
        Returns:
            Channel with lifted shadows
        """
        # Use different curves for different shadow depths
        l_norm = l_channel / 255.0
        
        # Logarithmic curve for deep shadows
        deep_shadow_curve = np.log1p(l_norm * 4) / np.log(5)
        
        # Power curve for medium shadows
        medium_shadow_curve = np.power(l_norm, 0.7)
        
        # Blend curves based on shadow depth
        lifted = l_norm.copy()
        
        # Apply deep shadow curve where shadows are strong
        deep_mask = shadow_mask > 0.7
        lifted[deep_mask] = deep_shadow_curve[deep_mask]
        
        # Apply medium shadow curve for moderate shadows
        medium_mask = (shadow_mask > 0.3) & (shadow_mask <= 0.7)
        lifted[medium_mask] = medium_shadow_curve[medium_mask]
        
        # Blend with original based on mask strength
        result = l_norm * (1 - shadow_mask) + lifted * shadow_mask
        
        return result * 255
    
    def enhance_shadow_contrast(self, l_channel: np.ndarray,
                               shadow_mask: np.ndarray) -> np.ndarray:
        """
        Enhance local contrast in recovered shadow regions
        
        Args:
            l_channel: Luminance channel with lifted shadows
            shadow_mask: Shadow mask
            
        Returns:
            Channel with enhanced shadow contrast
        """
        # Apply CLAHE only to shadow regions
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        l_uint8 = np.clip(l_channel, 0, 255).astype(np.uint8)
        l_enhanced = clahe.apply(l_uint8).astype(np.float32)
        
        # Blend based on shadow mask
        result = l_channel * (1 - shadow_mask * 0.5) + l_enhanced * shadow_mask * 0.5
        
        return result
    
    def compress_highlights(self, l_channel: np.ndarray,
                           highlight_mask: np.ndarray) -> np.ndarray:
        """
        Compress highlights to recover detail
        
        Args:
            l_channel: Luminance channel
            highlight_mask: Highlight mask
            
        Returns:
            Channel with compressed highlights
        """
        # Normalize to 0-1
        l_norm = l_channel / 255.0
        
        # Apply S-curve compression to highlights
        compressed = l_norm.copy()
        
        # Reinhard tone mapping for highlights
        bright_pixels = l_norm > 0.8
        if np.any(bright_pixels):
            x = l_norm[bright_pixels]
            # Compress using modified Reinhard operator
            compressed[bright_pixels] = x / (1 + x * 0.5)
            # Scale back to maintain overall brightness
            compressed[bright_pixels] = 0.8 + compressed[bright_pixels] * 0.2
        
        # Blend based on mask
        result = l_norm * (1 - highlight_mask) + compressed * highlight_mask
        
        return result * 255
    
    def recover_highlight_color(self, lab: np.ndarray,
                               highlight_mask: np.ndarray) -> np.ndarray:
        """
        Recover color information in highlight regions
        
        Args:
            lab: LAB image
            highlight_mask: Highlight mask
            
        Returns:
            LAB image with recovered highlight colors
        """
        # Boost A/B channels in highlight regions
        # This helps recover color that was lost to saturation
        
        a_channel = lab[:, :, 1].astype(np.float32)
        b_channel = lab[:, :, 2].astype(np.float32)
        
        # Center around neutral (128)
        a_centered = a_channel - 128
        b_centered = b_channel - 128
        
        # Boost color channels in highlights
        boost_factor = 1.5
        a_boosted = a_centered * (1 + highlight_mask * (boost_factor - 1))
        b_boosted = b_centered * (1 + highlight_mask * (boost_factor - 1))
        
        # Add back center
        lab[:, :, 1] = np.clip(a_boosted + 128, 0, 255).astype(np.uint8)
        lab[:, :, 2] = np.clip(b_boosted + 128, 0, 255).astype(np.uint8)
        
        return lab
    
    def detect_dark_disease_patterns(self, image: np.ndarray) -> np.ndarray:
        """
        Detect dark disease patterns (blight, necrosis) to protect during shadow recovery
        
        Args:
            image: RGB image
            
        Returns:
            Float mask (0-1) of disease regions
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Dark brown/black disease patterns
        lower_brown = np.array([0, 30, 10])
        upper_brown = np.array([20, 255, 60])
        brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
        
        # Very dark spots (necrosis)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, dark_mask = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)
        
        # Combine masks
        disease_mask = cv2.bitwise_or(brown_mask, dark_mask)
        
        # Clean up with morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_CLOSE, kernel)
        
        # Convert to float and smooth
        disease_mask = disease_mask.astype(np.float32) / 255.0
        disease_mask = cv2.GaussianBlur(disease_mask, (11, 11), 0)
        
        return disease_mask
    
    def detect_bright_disease_patterns(self, image: np.ndarray) -> np.ndarray:
        """
        Detect bright disease patterns (powdery mildew) to protect during highlight recovery
        
        Args:
            image: RGB image
            
        Returns:
            Float mask (0-1) of disease regions
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # White/gray powdery mildew
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 40, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Also check for low saturation bright areas
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        powdery_mask = (saturation < 30) & (value > 200)
        
        # Combine masks
        disease_mask = cv2.bitwise_or(white_mask, powdery_mask.astype(np.uint8) * 255)
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_CLOSE, kernel)
        
        # Convert to float and smooth
        disease_mask = disease_mask.astype(np.float32) / 255.0
        disease_mask = cv2.GaussianBlur(disease_mask, (11, 11), 0)
        
        return disease_mask