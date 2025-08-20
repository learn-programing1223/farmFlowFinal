"""
RGB-based Segmentation for Simple Backgrounds
Uses vegetation indices and color thresholds for fast segmentation
Always includes disease regions detected by DiseaseRegionDetector
"""

import numpy as np
import cv2
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class RGBSegmentation:
    """
    Fast RGB-based segmentation for simple backgrounds
    Handles 60-70% of cases in <20ms
    
    Key principle: Include ALL plant material (healthy AND diseased)
    """
    
    def __init__(self):
        """Initialize RGB segmentation"""
        # Vegetation index thresholds
        self.vari_threshold = -0.1  # Liberal threshold
        self.mgrvi_threshold = -0.05
        self.vndvi_threshold = -0.05
        
        # Color thresholds (inclusive of disease colors)
        self.green_hue_range = (25, 95)  # Healthy green
        self.brown_hue_range = (0, 30)   # Disease browns
        self.yellow_hue_range = (20, 40) # Chlorotic yellows
        
        # Performance stats
        self.stats = {
            'processed': 0,
            'avg_time_ms': 0
        }
    
    def segment(self, image: np.ndarray, 
                disease_mask: Optional[np.ndarray] = None,
                vegetation_indices: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Perform RGB-based segmentation
        
        Args:
            image: RGB image
            disease_mask: Pre-detected disease regions (MUST be included)
            vegetation_indices: Pre-computed indices from pipeline
            
        Returns:
            Tuple of (segmentation_mask, info)
        """
        import time
        start_time = time.time()
        
        h, w = image.shape[:2]
        info = {'method': 'rgb', 'indices_used': False}
        
        # Initialize mask
        plant_mask = np.zeros((h, w), dtype=np.uint8)
        
        # 1. Use vegetation indices if available
        if vegetation_indices is not None:
            indices_mask = self.segment_with_indices(vegetation_indices, (h, w))
            plant_mask = cv2.bitwise_or(plant_mask, indices_mask)
            info['indices_used'] = True
        else:
            # Calculate indices if not provided
            indices_mask = self.calculate_and_segment_indices(image)
            plant_mask = cv2.bitwise_or(plant_mask, indices_mask)
        
        # 2. Color-based segmentation
        color_mask = self.segment_by_color(image)
        plant_mask = cv2.bitwise_or(plant_mask, color_mask)
        
        # 3. HSV-based vegetation detection
        hsv_mask = self.segment_by_hsv(image)
        plant_mask = cv2.bitwise_or(plant_mask, hsv_mask)
        
        # 4. CRITICAL: Always include disease mask
        if disease_mask is not None:
            # Dilate disease mask slightly to include boundaries
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            disease_dilated = cv2.dilate(disease_mask, kernel, iterations=1)
            plant_mask = cv2.bitwise_or(plant_mask, disease_dilated)
            info['disease_included'] = True
        else:
            info['disease_included'] = False
        
        # 5. Refine mask
        plant_mask = self.refine_mask(plant_mask)
        
        # 6. Fill holes (disease regions might create holes)
        plant_mask = self.fill_holes(plant_mask)
        
        # Calculate coverage
        info['coverage'] = np.sum(plant_mask > 0) / (h * w)
        
        # Update stats
        elapsed_ms = (time.time() - start_time) * 1000
        self.update_stats(elapsed_ms)
        
        logger.debug(f"RGB segmentation in {elapsed_ms:.1f}ms, coverage: {info['coverage']:.2%}")
        
        return plant_mask, info
    
    def segment_with_indices(self, indices: Dict, shape: Tuple[int, int]) -> np.ndarray:
        """
        Segment using pre-computed vegetation indices
        
        Args:
            indices: Dictionary with VARI, MGRVI, vNDVI
            shape: Output shape (h, w)
            
        Returns:
            Binary mask
        """
        h, w = shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # VARI (Visible Atmospherically Resistant Index)
        if 'vari' in indices:
            vari = indices['vari']
            # Ensure correct shape
            if vari.shape[:2] == (h, w):
                vari_mask = (vari > self.vari_threshold).astype(np.uint8) * 255
                mask = cv2.bitwise_or(mask, vari_mask)
        
        # MGRVI (Modified Green Red Vegetation Index)
        if 'mgrvi' in indices:
            mgrvi = indices['mgrvi']
            if mgrvi.shape[:2] == (h, w):
                mgrvi_mask = (mgrvi > self.mgrvi_threshold).astype(np.uint8) * 255
                mask = cv2.bitwise_or(mask, mgrvi_mask)
        
        # vNDVI (visible Normalized Difference Vegetation Index)
        if 'vndvi' in indices:
            vndvi = indices['vndvi']
            if vndvi.shape[:2] == (h, w):
                vndvi_mask = (vndvi > self.vndvi_threshold).astype(np.uint8) * 255
                mask = cv2.bitwise_or(mask, vndvi_mask)
        
        return mask
    
    def calculate_and_segment_indices(self, image: np.ndarray) -> np.ndarray:
        """
        Calculate vegetation indices and segment
        
        Args:
            image: RGB image
            
        Returns:
            Binary mask
        """
        # Convert to float
        img_float = image.astype(np.float32) / 255.0
        
        # Extract channels
        r = img_float[:, :, 0]
        g = img_float[:, :, 1]
        b = img_float[:, :, 2]
        
        eps = 1e-10
        
        # VARI = (G - R) / (G + R - B + eps)
        denominator = g + r - b + eps
        vari = np.where(denominator != 0, (g - r) / denominator, 0)
        vari_mask = (vari > self.vari_threshold).astype(np.uint8) * 255
        
        # MGRVI = (G² - R²) / (G² + R² + eps)
        g_squared = g ** 2
        r_squared = r ** 2
        denominator = g_squared + r_squared + eps
        mgrvi = (g_squared - r_squared) / denominator
        mgrvi_mask = (mgrvi > self.mgrvi_threshold).astype(np.uint8) * 255
        
        # vNDVI = (G - R) / (G + R + eps)
        denominator = g + r + eps
        vndvi = (g - r) / denominator
        vndvi_mask = (vndvi > self.vndvi_threshold).astype(np.uint8) * 255
        
        # Combine all indices
        combined = cv2.bitwise_or(vari_mask, mgrvi_mask)
        combined = cv2.bitwise_or(combined, vndvi_mask)
        
        return combined
    
    def segment_by_color(self, image: np.ndarray) -> np.ndarray:
        """
        Segment by direct color thresholds
        Includes disease colors (brown, yellow)
        
        Args:
            image: RGB image
            
        Returns:
            Binary mask
        """
        # Create mask for plant-like colors
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Green vegetation (healthy)
        # High green, moderate red and blue
        green_mask = (
            (image[:, :, 1] > image[:, :, 0] * 1.1) &  # G > R
            (image[:, :, 1] > image[:, :, 2] * 1.1) &  # G > B
            (image[:, :, 1] > 30)  # Minimum green value
        ).astype(np.uint8) * 255
        
        mask = cv2.bitwise_or(mask, green_mask)
        
        # Brown vegetation (diseased)
        # R and G similar, B lower
        brown_mask = (
            (np.abs(image[:, :, 0].astype(np.int16) - image[:, :, 1]) < 50) &
            (image[:, :, 2] < image[:, :, 0] * 0.8) &
            (image[:, :, 0] > 30) & (image[:, :, 0] < 150)
        ).astype(np.uint8) * 255
        
        mask = cv2.bitwise_or(mask, brown_mask)
        
        # Yellow vegetation (chlorotic)
        # High R and G, low B
        yellow_mask = (
            (image[:, :, 0] > 100) &
            (image[:, :, 1] > 100) &
            (image[:, :, 2] < 100) &
            (np.abs(image[:, :, 0].astype(np.int16) - image[:, :, 1]) < 30)
        ).astype(np.uint8) * 255
        
        mask = cv2.bitwise_or(mask, yellow_mask)
        
        return mask
    
    def segment_by_hsv(self, image: np.ndarray) -> np.ndarray:
        """
        Segment using HSV color space
        More robust to lighting variations
        
        Args:
            image: RGB image
            
        Returns:
            Binary mask
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        masks = []
        
        # Green vegetation (healthy)
        lower_green = np.array([25, 20, 20])
        upper_green = np.array([95, 255, 255])
        masks.append(cv2.inRange(hsv, lower_green, upper_green))
        
        # Brown vegetation (diseased)
        lower_brown = np.array([0, 20, 20])
        upper_brown = np.array([30, 255, 200])
        masks.append(cv2.inRange(hsv, lower_brown, upper_brown))
        
        # Yellow vegetation (chlorotic)
        lower_yellow = np.array([20, 30, 50])
        upper_yellow = np.array([40, 255, 255])
        masks.append(cv2.inRange(hsv, lower_yellow, upper_yellow))
        
        # Dark green (shadows on leaves)
        lower_dark = np.array([40, 10, 10])
        upper_dark = np.array([80, 255, 100])
        masks.append(cv2.inRange(hsv, lower_dark, upper_dark))
        
        # White/gray (powdery mildew on leaves)
        # Low saturation but on greenish background
        lower_white = np.array([0, 0, 100])
        upper_white = np.array([180, 30, 255])
        
        # Check if white areas are near green (likely mildew on leaf)
        white_candidate = cv2.inRange(hsv, lower_white, upper_white)
        
        # Dilate green mask to find nearby green
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        green_dilated = cv2.dilate(masks[0], kernel, iterations=2)
        
        # White near green is likely powdery mildew
        white_on_plant = cv2.bitwise_and(white_candidate, green_dilated)
        masks.append(white_on_plant)
        
        # Combine all masks
        combined = np.zeros_like(masks[0])
        for mask in masks:
            combined = cv2.bitwise_or(combined, mask)
        
        return combined
    
    def refine_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Refine segmentation mask
        
        Args:
            mask: Binary mask
            
        Returns:
            Refined mask
        """
        # Remove small noise
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
        
        # Close gaps
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        closed = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)
        
        # Smooth boundaries
        kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        smoothed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel_smooth)
        smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel_smooth)
        
        return smoothed
    
    def fill_holes(self, mask: np.ndarray) -> np.ndarray:
        """
        Fill holes in mask (important for diseased areas)
        
        Args:
            mask: Binary mask
            
        Returns:
            Mask with holes filled
        """
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create filled mask
        filled = np.zeros_like(mask)
        
        # Draw filled contours
        cv2.drawContours(filled, contours, -1, 255, -1)
        
        return filled
    
    def update_stats(self, time_ms: float):
        """Update segmentation statistics"""
        n = self.stats['processed'] + 1
        self.stats['processed'] = n
        self.stats['avg_time_ms'] = (self.stats['avg_time_ms'] * (n-1) + time_ms) / n
    
    def get_stats(self) -> Dict:
        """Get segmentation statistics"""
        return self.stats.copy()
    
    def evaluate_quality(self, mask: np.ndarray) -> float:
        """
        Evaluate segmentation quality
        
        Args:
            mask: Binary mask
            
        Returns:
            Quality score (0-1)
        """
        h, w = mask.shape
        
        # Check coverage (not too little, not too much)
        coverage = np.sum(mask > 0) / (h * w)
        
        if coverage < 0.05:  # Too little
            return 0.2
        elif coverage > 0.95:  # Too much (likely failed)
            return 0.3
        
        # Check connectivity (plant should be connected)
        num_labels, _ = cv2.connectedComponents(mask)
        
        # Ideal: 1-3 connected components
        if num_labels <= 4:
            connectivity_score = 1.0
        else:
            connectivity_score = 4.0 / num_labels
        
        # Check shape regularity
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            # Get largest contour
            largest = max(contours, key=cv2.contourArea)
            
            # Calculate solidity (area / convex hull area)
            hull = cv2.convexHull(largest)
            hull_area = cv2.contourArea(hull)
            contour_area = cv2.contourArea(largest)
            
            if hull_area > 0:
                solidity = contour_area / hull_area
            else:
                solidity = 0
        else:
            solidity = 0
        
        # Combine scores
        quality = (connectivity_score * 0.4 + 
                  solidity * 0.3 + 
                  min(coverage * 2, 1.0) * 0.3)
        
        return quality