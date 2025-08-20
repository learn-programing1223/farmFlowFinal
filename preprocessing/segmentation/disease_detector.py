"""
Disease Region Detector for Segmentation
Identifies ALL potential disease regions BEFORE segmentation to ensure preservation
Critical component: Better to over-detect than miss disease
"""

import numpy as np
import cv2
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class DiseaseRegionDetector:
    """
    Detects all potential disease regions in plant images
    Philosophy: Conservative detection - include anything that might be disease
    
    Detects:
    - Brown/necrotic regions (blight, bacterial spot)
    - Yellow/chlorotic areas (virus, nutrient deficiency)
    - White/gray patches (powdery mildew)
    - Dark spots and lesions (various diseases)
    - Texture anomalies (any disease pattern)
    """
    
    def __init__(self, sensitivity: float = 0.8):
        """
        Initialize disease detector
        
        Args:
            sensitivity: Detection sensitivity (0.0-1.0), higher = more conservative
        """
        self.sensitivity = sensitivity
        
        # Adjust thresholds based on sensitivity
        self.adjust_thresholds()
        
        # Performance stats
        self.stats = {
            'detections': 0,
            'avg_time_ms': 0,
            'avg_coverage': 0
        }
    
    def adjust_thresholds(self):
        """Adjust detection thresholds based on sensitivity"""
        # Base thresholds (at sensitivity = 0.5)
        base_brown_threshold = 0.05
        base_yellow_threshold = 0.05
        base_white_threshold = 0.05
        base_texture_threshold = 20
        
        # Adjust based on sensitivity
        factor = 2.0 - self.sensitivity  # Higher sensitivity = lower thresholds
        
        self.brown_threshold = base_brown_threshold * factor
        self.yellow_threshold = base_yellow_threshold * factor
        self.white_threshold = base_white_threshold * factor
        self.texture_threshold = base_texture_threshold * factor
    
    def detect(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Main detection method - finds ALL disease regions
        
        Args:
            image: RGB image
            
        Returns:
            Tuple of (disease_mask, detection_info)
        """
        import time
        start_time = time.time()
        
        h, w = image.shape[:2]
        
        # Initialize combined mask
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Detection info
        info = {
            'brown_regions': False,
            'yellow_regions': False,
            'white_regions': False,
            'dark_spots': False,
            'texture_anomalies': False,
            'total_coverage': 0
        }
        
        # 1. Detect brown/necrotic regions (blight, bacterial spot)
        brown_mask = self.detect_brown_necrotic(image)
        if np.any(brown_mask):
            combined_mask = cv2.bitwise_or(combined_mask, brown_mask)
            info['brown_regions'] = True
        
        # 2. Detect yellow/chlorotic areas (virus, deficiency)
        yellow_mask = self.detect_yellow_chlorotic(image)
        if np.any(yellow_mask):
            combined_mask = cv2.bitwise_or(combined_mask, yellow_mask)
            info['yellow_regions'] = True
        
        # 3. Detect white/gray patches (powdery mildew)
        white_mask = self.detect_white_gray(image)
        if np.any(white_mask):
            combined_mask = cv2.bitwise_or(combined_mask, white_mask)
            info['white_regions'] = True
        
        # 4. Detect dark spots and lesions
        dark_mask = self.detect_dark_spots(image)
        if np.any(dark_mask):
            combined_mask = cv2.bitwise_or(combined_mask, dark_mask)
            info['dark_spots'] = True
        
        # 5. Detect texture anomalies
        texture_mask = self.detect_texture_anomalies(image)
        if np.any(texture_mask):
            combined_mask = cv2.bitwise_or(combined_mask, texture_mask)
            info['texture_anomalies'] = True
        
        # 6. Apply morphological operations to clean up
        combined_mask = self.refine_mask(combined_mask)
        
        # Calculate coverage
        info['total_coverage'] = np.sum(combined_mask > 0) / (h * w)
        
        # Update stats
        elapsed_ms = (time.time() - start_time) * 1000
        self.update_stats(elapsed_ms, info['total_coverage'])
        
        logger.debug(f"Disease detection in {elapsed_ms:.1f}ms, coverage: {info['total_coverage']:.2%}")
        
        return combined_mask, info
    
    def detect_brown_necrotic(self, image: np.ndarray) -> np.ndarray:
        """
        Detect brown and necrotic regions (blight, bacterial diseases)
        
        Args:
            image: RGB image
            
        Returns:
            Binary mask of brown regions
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Multiple brown ranges for comprehensive detection
        masks = []
        
        # Dark brown (necrotic tissue)
        lower1 = np.array([0, 30, 20])
        upper1 = np.array([15, 255, 100])
        masks.append(cv2.inRange(hsv, lower1, upper1))
        
        # Medium brown (advancing disease)
        lower2 = np.array([10, 40, 40])
        upper2 = np.array([20, 200, 150])
        masks.append(cv2.inRange(hsv, lower2, upper2))
        
        # Light brown (early disease)
        lower3 = np.array([15, 30, 60])
        upper3 = np.array([25, 150, 180])
        masks.append(cv2.inRange(hsv, lower3, upper3))
        
        # Reddish-brown (some blights)
        lower4 = np.array([0, 50, 50])
        upper4 = np.array([10, 200, 150])
        masks.append(cv2.inRange(hsv, lower4, upper4))
        
        # Combine all brown masks
        brown_mask = np.zeros_like(masks[0])
        for mask in masks:
            brown_mask = cv2.bitwise_or(brown_mask, mask)
        
        # Check if significant brown regions exist
        coverage = np.sum(brown_mask > 0) / brown_mask.size
        if coverage < self.brown_threshold:
            return np.zeros_like(brown_mask)
        
        return brown_mask
    
    def detect_yellow_chlorotic(self, image: np.ndarray) -> np.ndarray:
        """
        Detect yellow and chlorotic areas (virus, nutrient deficiency)
        
        Args:
            image: RGB image
            
        Returns:
            Binary mask of yellow regions
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        masks = []
        
        # Bright yellow (virus symptoms)
        lower1 = np.array([20, 40, 100])
        upper1 = np.array([35, 255, 255])
        masks.append(cv2.inRange(hsv, lower1, upper1))
        
        # Pale yellow (chlorosis)
        lower2 = np.array([25, 20, 120])
        upper2 = np.array([40, 100, 255])
        masks.append(cv2.inRange(hsv, lower2, upper2))
        
        # Yellow-green (early chlorosis)
        lower3 = np.array([35, 30, 80])
        upper3 = np.array([50, 150, 200])
        masks.append(cv2.inRange(hsv, lower3, upper3))
        
        # Orange-yellow (some diseases)
        lower4 = np.array([15, 50, 100])
        upper4 = np.array([25, 255, 255])
        masks.append(cv2.inRange(hsv, lower4, upper4))
        
        # Combine yellow masks
        yellow_mask = np.zeros_like(masks[0])
        for mask in masks:
            yellow_mask = cv2.bitwise_or(yellow_mask, mask)
        
        # Also check for low chlorophyll (high red/green ratio)
        rgb_ratio_mask = self.detect_low_chlorophyll(image)
        yellow_mask = cv2.bitwise_or(yellow_mask, rgb_ratio_mask)
        
        # Check significance
        coverage = np.sum(yellow_mask > 0) / yellow_mask.size
        if coverage < self.yellow_threshold:
            return np.zeros_like(yellow_mask)
        
        return yellow_mask
    
    def detect_white_gray(self, image: np.ndarray) -> np.ndarray:
        """
        Detect white and gray patches (powdery mildew)
        
        Args:
            image: RGB image
            
        Returns:
            Binary mask of white/gray regions
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        masks = []
        
        # White patches (powdery mildew)
        lower1 = np.array([0, 0, 180])
        upper1 = np.array([180, 40, 255])
        masks.append(cv2.inRange(hsv, lower1, upper1))
        
        # Light gray (dusty mildew)
        lower2 = np.array([0, 0, 120])
        upper2 = np.array([180, 30, 200])
        masks.append(cv2.inRange(hsv, lower2, upper2))
        
        # Off-white with any hue
        lower3 = np.array([0, 0, 200])
        upper3 = np.array([180, 20, 255])
        masks.append(cv2.inRange(hsv, lower3, upper3))
        
        # Combine white masks
        white_mask = np.zeros_like(masks[0])
        for mask in masks:
            white_mask = cv2.bitwise_or(white_mask, mask)
        
        # Additional texture check for powdery appearance
        texture_check = self.detect_powdery_texture(image)
        white_mask = cv2.bitwise_and(white_mask, texture_check)
        
        # Check significance
        coverage = np.sum(white_mask > 0) / white_mask.size
        if coverage < self.white_threshold:
            return np.zeros_like(white_mask)
        
        return white_mask
    
    def detect_dark_spots(self, image: np.ndarray) -> np.ndarray:
        """
        Detect dark spots and lesions (various diseases)
        
        Args:
            image: RGB image
            
        Returns:
            Binary mask of dark spots
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Adaptive threshold to find dark regions
        mean = np.mean(gray)
        
        # Dark spots are significantly darker than mean
        threshold = mean * 0.4  # Adjust based on sensitivity
        _, dark_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        
        # Remove very large dark regions (likely shadows, not spots)
        # Keep only spot-like features
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        opened = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel_open)
        
        # Spots are small dark regions
        spots = cv2.subtract(dark_mask, opened)
        
        # Also detect using blob detection for circular spots
        circular_spots = self.detect_circular_spots(gray)
        
        # Combine
        dark_spots = cv2.bitwise_or(spots, circular_spots)
        
        return dark_spots
    
    def detect_texture_anomalies(self, image: np.ndarray) -> np.ndarray:
        """
        Detect texture anomalies that might indicate disease
        
        Args:
            image: RGB image
            
        Returns:
            Binary mask of texture anomalies
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Calculate local variance to detect texture changes
        kernel_size = 5
        mean = cv2.blur(gray, (kernel_size, kernel_size))
        sq_mean = cv2.blur(gray**2, (kernel_size, kernel_size))
        variance = sq_mean - mean**2
        
        # High variance indicates texture anomaly
        _, texture_mask = cv2.threshold(variance, self.texture_threshold, 255, cv2.THRESH_BINARY)
        texture_mask = texture_mask.astype(np.uint8)
        
        # Also check for mottled patterns (mosaic virus)
        mottled = self.detect_mottled_pattern(image)
        
        # Combine
        anomaly_mask = cv2.bitwise_or(texture_mask, mottled)
        
        return anomaly_mask
    
    def detect_low_chlorophyll(self, image: np.ndarray) -> np.ndarray:
        """
        Detect areas with low chlorophyll (yellowing)
        
        Args:
            image: RGB image
            
        Returns:
            Binary mask
        """
        # Calculate green dominance
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        
        # Low chlorophyll: green is not dominant
        green_dominance = (g.astype(np.float32) / (r + g + b + 1e-6))
        low_chlorophyll = green_dominance < 0.35
        
        # Also check red/green ratio
        rg_ratio = r.astype(np.float32) / (g + 1e-6)
        high_rg = rg_ratio > 0.9
        
        # Combine conditions
        mask = np.logical_or(low_chlorophyll, high_rg)
        
        return (mask * 255).astype(np.uint8)
    
    def detect_powdery_texture(self, image: np.ndarray) -> np.ndarray:
        """
        Detect powdery/dusty texture (powdery mildew)
        
        Args:
            image: RGB image
            
        Returns:
            Binary mask
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Powdery texture has specific frequency characteristics
        # Use Laplacian to detect fine texture
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        # Powdery areas have uniform fine texture
        kernel_size = 9
        mean_lap = cv2.blur(np.abs(laplacian), (kernel_size, kernel_size))
        
        # Threshold for powdery texture
        threshold = np.percentile(mean_lap, 70)
        mask = (mean_lap > threshold).astype(np.uint8) * 255
        
        return mask
    
    def detect_circular_spots(self, gray: np.ndarray) -> np.ndarray:
        """
        Detect circular spots (leaf spot disease)
        
        Args:
            gray: Grayscale image
            
        Returns:
            Binary mask of circular spots
        """
        # Use HoughCircles to detect circular patterns
        circles_mask = np.zeros_like(gray)
        
        # Blur for better circle detection
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Detect circles
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=30,
            minRadius=3,
            maxRadius=30
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                # Draw filled circles
                cv2.circle(circles_mask, (circle[0], circle[1]), circle[2], 255, -1)
        
        return circles_mask
    
    def detect_mottled_pattern(self, image: np.ndarray) -> np.ndarray:
        """
        Detect mottled patterns (mosaic virus)
        
        Args:
            image: RGB image
            
        Returns:
            Binary mask
        """
        # Convert to LAB for better color separation
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]
        
        # Mottled patterns have alternating light/dark regions
        # Use local histogram analysis
        kernel_size = 15
        
        # Calculate local mean
        local_mean = cv2.blur(l_channel, (kernel_size, kernel_size))
        
        # Calculate deviation from local mean
        deviation = np.abs(l_channel.astype(np.float32) - local_mean.astype(np.float32))
        
        # High deviation in regular patterns indicates mottling
        # Use morphological gradient to detect edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        gradient = cv2.morphologyEx(deviation.astype(np.uint8), cv2.MORPH_GRADIENT, kernel)
        
        # Threshold
        _, mottled_mask = cv2.threshold(gradient, 10, 255, cv2.THRESH_BINARY)
        
        return mottled_mask
    
    def refine_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Refine the disease mask with morphological operations
        
        Args:
            mask: Binary mask
            
        Returns:
            Refined mask
        """
        # Remove small noise
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
        
        # Close small gaps
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)
        
        # Dilate slightly to include boundaries
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated = cv2.dilate(closed, kernel_dilate, iterations=1)
        
        return dilated
    
    def update_stats(self, time_ms: float, coverage: float):
        """Update detection statistics"""
        n = self.stats['detections'] + 1
        self.stats['detections'] = n
        
        # Update averages
        self.stats['avg_time_ms'] = (self.stats['avg_time_ms'] * (n-1) + time_ms) / n
        self.stats['avg_coverage'] = (self.stats['avg_coverage'] * (n-1) + coverage) / n
    
    def get_stats(self) -> Dict:
        """Get detection statistics"""
        return self.stats.copy()