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
        # Higher thresholds = more selective (less false positives)
        base_brown_threshold = 0.25  # Require 25% coverage to trigger
        base_yellow_threshold = 0.20  # Require 20% coverage
        base_white_threshold = 0.30  # Require 30% coverage
        base_texture_threshold = 100  # Much higher for less sensitivity
        
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
        
        # Create healthy mask to exclude from detection
        healthy_mask = self.create_healthy_mask(image)
        
        # 1. Detect brown/necrotic regions (blight, bacterial spot)
        brown_mask = self.detect_brown_necrotic(image)
        # Exclude healthy regions
        brown_mask = cv2.bitwise_and(brown_mask, cv2.bitwise_not(healthy_mask))
        if np.sum(brown_mask > 0) > 100:  # Minimum pixel count
            combined_mask = cv2.bitwise_or(combined_mask, brown_mask)
            info['brown_regions'] = True
        
        # 2. Detect yellow/chlorotic areas (virus, deficiency)
        yellow_mask = self.detect_yellow_chlorotic(image)
        # Exclude healthy regions
        yellow_mask = cv2.bitwise_and(yellow_mask, cv2.bitwise_not(healthy_mask))
        if np.sum(yellow_mask > 0) > 100:  # Minimum pixel count
            combined_mask = cv2.bitwise_or(combined_mask, yellow_mask)
            info['yellow_regions'] = True
        
        # 3. Detect white/gray patches (powdery mildew)
        white_mask = self.detect_white_gray(image)
        # Exclude healthy regions
        white_mask = cv2.bitwise_and(white_mask, cv2.bitwise_not(healthy_mask))
        if np.sum(white_mask > 0) > 100:  # Minimum pixel count
            combined_mask = cv2.bitwise_or(combined_mask, white_mask)
            info['white_regions'] = True
        
        # 4. Detect dark spots and lesions
        dark_mask = self.detect_dark_spots(image)
        # Exclude healthy regions from dark spots too
        dark_mask = cv2.bitwise_and(dark_mask, cv2.bitwise_not(healthy_mask))
        if np.sum(dark_mask > 0) > 100:  # Minimum pixel count
            combined_mask = cv2.bitwise_or(combined_mask, dark_mask)
            info['dark_spots'] = True
        
        # 5. Skip texture anomalies for now - too sensitive
        # texture_mask = self.detect_texture_anomalies(image)
        info['texture_anomalies'] = False
        
        # 6. Apply morphological operations to clean up
        combined_mask = self.refine_mask(combined_mask)
        
        # 7. Remove small isolated regions (noise)
        combined_mask = self.remove_small_regions(combined_mask, min_area=50)
        
        # Calculate coverage
        info['total_coverage'] = np.sum(combined_mask > 0) / (h * w)
        
        # Update stats
        elapsed_ms = (time.time() - start_time) * 1000
        self.update_stats(elapsed_ms, info['total_coverage'])
        
        logger.debug(f"Disease detection in {elapsed_ms:.1f}ms, coverage: {info['total_coverage']:.2%}")
        
        return combined_mask, info
    
    def create_healthy_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Create mask of healthy green tissue to exclude from disease detection
        
        Args:
            image: RGB image
            
        Returns:
            Binary mask of healthy regions (255 = healthy, 0 = not healthy)
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Healthy green range (based on actual synthetic leaf stats: Hue 29-86)
        # Primary healthy green
        lower_green = np.array([40, 50, 70])
        upper_green = np.array([85, 255, 255])
        healthy_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Also include yellow-greens that are still healthy
        lower_yellow_green = np.array([30, 40, 60])
        upper_yellow_green = np.array([45, 200, 200])
        dark_green_mask = cv2.inRange(hsv, lower_yellow_green, upper_yellow_green)
        
        # Combine healthy masks
        healthy_combined = cv2.bitwise_or(healthy_mask, dark_green_mask)
        
        # Dilate slightly to include boundaries
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        healthy_combined = cv2.dilate(healthy_combined, kernel, iterations=1)
        
        return healthy_combined
    
    def remove_small_regions(self, mask: np.ndarray, min_area: int = 50) -> np.ndarray:
        """
        Remove small isolated regions from mask (noise removal)
        
        Args:
            mask: Binary mask
            min_area: Minimum contour area to keep
            
        Returns:
            Cleaned mask
        """
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create clean mask
        clean_mask = np.zeros_like(mask)
        
        # Keep only contours above minimum area
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                cv2.drawContours(clean_mask, [contour], -1, 255, -1)
        
        return clean_mask
    
    def detect_brown_necrotic(self, image: np.ndarray) -> np.ndarray:
        """
        Detect brown and necrotic regions (blight, bacterial diseases)
        
        Args:
            image: RGB image
            
        Returns:
            Binary mask of brown regions
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # More selective brown ranges for true disease
        masks = []
        
        # Dark spots with low value (true necrosis)
        # Looking for Hue 0-25 with low Value (<60)
        lower1 = np.array([0, 50, 10])
        upper1 = np.array([25, 255, 60])
        masks.append(cv2.inRange(hsv, lower1, upper1))
        
        # Brown diseased areas (not green)
        lower2 = np.array([10, 60, 20])
        upper2 = np.array([20, 200, 80])
        masks.append(cv2.inRange(hsv, lower2, upper2))
        
        # Combine all brown masks
        brown_mask = np.zeros_like(masks[0])
        for mask in masks:
            brown_mask = cv2.bitwise_or(brown_mask, mask)
        
        # Return the actual detected regions (no threshold check here)
        # The threshold should be used differently
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
        
        # True yellow (virus symptoms, not green)
        lower1 = np.array([22, 80, 150])
        upper1 = np.array([32, 255, 255])
        masks.append(cv2.inRange(hsv, lower1, upper1))
        
        # Pale yellow/chlorotic (nutrient deficiency)
        lower2 = np.array([25, 40, 180])
        upper2 = np.array([35, 120, 255])
        masks.append(cv2.inRange(hsv, lower2, upper2))
        
        # NOTE: Removed yellow-green range as it catches healthy leaves
        # NOTE: Removed orange overlap with brown detection
        
        # Combine yellow masks
        yellow_mask = np.zeros_like(masks[0])
        for mask in masks:
            yellow_mask = cv2.bitwise_or(yellow_mask, mask)
        
        # Also check for low chlorophyll (high red/green ratio)
        rgb_ratio_mask = self.detect_low_chlorophyll(image)
        yellow_mask = cv2.bitwise_or(yellow_mask, rgb_ratio_mask)
        
        # Return the actual detected regions
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
        
        # True white patches (powdery mildew)
        lower1 = np.array([0, 0, 200])  # Higher value threshold
        upper1 = np.array([180, 30, 255])  # Lower saturation for true white
        masks.append(cv2.inRange(hsv, lower1, upper1))
        
        # Light gray powdery coating
        lower2 = np.array([0, 0, 160])
        upper2 = np.array([180, 25, 210])
        masks.append(cv2.inRange(hsv, lower2, upper2))
        
        # Very white patches (severe mildew)
        lower3 = np.array([0, 0, 220])
        upper3 = np.array([180, 15, 255])
        masks.append(cv2.inRange(hsv, lower3, upper3))
        
        # Combine white masks
        white_mask = np.zeros_like(masks[0])
        for mask in masks:
            white_mask = cv2.bitwise_or(white_mask, mask)
        
        # Additional texture check for powdery appearance
        texture_check = self.detect_powdery_texture(image)
        white_mask = cv2.bitwise_and(white_mask, texture_check)
        
        # Return the actual detected regions
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
        
        # Use adaptive threshold based on image statistics
        mean_variance = np.mean(variance)
        std_variance = np.std(variance)
        
        # Only flag extreme anomalies (2 standard deviations above mean)
        adaptive_threshold = mean_variance + 2 * std_variance
        adaptive_threshold = max(adaptive_threshold, self.texture_threshold)
        
        _, texture_mask = cv2.threshold(variance, adaptive_threshold, 255, cv2.THRESH_BINARY)
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
        
        # Low chlorophyll: green is very low (not just non-dominant)
        green_dominance = (g.astype(np.float32) / (r + g + b + 1e-6))
        low_chlorophyll = green_dominance < 0.28  # More selective
        
        # Also check red/green ratio (higher threshold)
        rg_ratio = r.astype(np.float32) / (g + 1e-6)
        high_rg = rg_ratio > 1.2  # More selective
        
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