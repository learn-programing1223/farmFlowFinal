"""
Local Adaptive Processing for Region-Specific Enhancement
Processes different image regions with different parameters based on local conditions
"""

import numpy as np
import cv2
from typing import Tuple, Dict, List
import logging

logger = logging.getLogger(__name__)


class LocalAdaptiveProcessor:
    """
    Grid-based adaptive processing for handling varied lighting within single image
    Common in field photos where leaves have different orientations to sun
    """
    
    def __init__(self, grid_size: Tuple[int, int] = (4, 4)):
        """
        Initialize local adaptive processor
        
        Args:
            grid_size: Grid divisions (rows, cols) for local processing
        """
        self.grid_size = grid_size
        self.blend_kernel_size = 31  # For smooth transitions between regions
        
    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Apply local adaptive processing
        
        Args:
            image: Input RGB image
            
        Returns:
            Locally processed image with smooth transitions
        """
        h, w = image.shape[:2]
        rows, cols = self.grid_size
        
        # Calculate region dimensions
        region_h = h // rows
        region_w = w // cols
        
        # Analyze each region
        region_params = self.analyze_regions(image, region_h, region_w)
        
        # Process each region with appropriate parameters
        processed_regions = self.process_regions(image, region_params, region_h, region_w)
        
        # Blend regions smoothly
        result = self.blend_regions(processed_regions, region_h, region_w)
        
        return result
    
    def analyze_regions(self, image: np.ndarray, 
                        region_h: int, region_w: int) -> List[List[Dict]]:
        """
        Analyze each grid region to determine optimal parameters
        
        Args:
            image: Input image
            region_h: Height of each region
            region_w: Width of each region
            
        Returns:
            Grid of parameters for each region
        """
        rows, cols = self.grid_size
        region_params = []
        
        for r in range(rows):
            row_params = []
            for c in range(cols):
                # Extract region
                y_start = r * region_h
                y_end = min((r + 1) * region_h, image.shape[0])
                x_start = c * region_w
                x_end = min((c + 1) * region_w, image.shape[1])
                
                region = image[y_start:y_end, x_start:x_end]
                
                # Analyze region characteristics
                params = self.analyze_single_region(region)
                params['bounds'] = (y_start, y_end, x_start, x_end)
                
                row_params.append(params)
            
            region_params.append(row_params)
        
        return region_params
    
    def analyze_single_region(self, region: np.ndarray) -> Dict:
        """
        Analyze a single region to determine processing parameters
        
        Args:
            region: Image region to analyze
            
        Returns:
            Dictionary of processing parameters
        """
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        
        # Calculate statistics
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        
        # Calculate histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hist_norm = hist / (hist.sum() + 1e-6)
        
        # Determine percentiles
        cumsum = np.cumsum(hist_norm)
        p10 = np.searchsorted(cumsum, 0.10)
        p90 = np.searchsorted(cumsum, 0.90)
        
        # Classify region type
        if mean_val > 200:
            region_type = 'overexposed'
            gamma = 0.6
            clahe_clip = 2.0
            enhance_strength = 0.3
        elif mean_val < 50:
            region_type = 'underexposed'
            gamma = 1.8
            clahe_clip = 4.0
            enhance_strength = 0.7
        elif std_val > 60:
            region_type = 'high_contrast'
            gamma = 1.0
            clahe_clip = 3.0
            enhance_strength = 0.5
        else:
            region_type = 'normal'
            gamma = 1.0
            clahe_clip = 2.5
            enhance_strength = 0.4
        
        # Check for disease-like patterns (important to preserve)
        has_disease = self.detect_disease_likelihood(region)
        
        if has_disease:
            # Be more conservative with disease regions
            enhance_strength *= 0.7
            clahe_clip = min(clahe_clip, 3.0)
        
        return {
            'type': region_type,
            'mean': mean_val,
            'std': std_val,
            'gamma': gamma,
            'clahe_clip': clahe_clip,
            'enhance_strength': enhance_strength,
            'dynamic_range': p90 - p10,
            'has_disease': has_disease
        }
    
    def detect_disease_likelihood(self, region: np.ndarray) -> bool:
        """
        Quick check if region likely contains disease patterns
        
        Args:
            region: Image region
            
        Returns:
            True if disease patterns likely present
        """
        # Convert to HSV for color analysis
        hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
        
        # Check for disease-indicative colors
        # Brown spots
        brown_mask = cv2.inRange(hsv, np.array([10, 50, 20]), np.array([20, 255, 150]))
        brown_ratio = np.sum(brown_mask > 0) / brown_mask.size
        
        # Yellow areas (chlorosis)
        yellow_mask = cv2.inRange(hsv, np.array([20, 50, 100]), np.array([30, 255, 255]))
        yellow_ratio = np.sum(yellow_mask > 0) / yellow_mask.size
        
        # White/gray (powdery mildew)
        white_mask = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 30, 255]))
        white_ratio = np.sum(white_mask > 0) / white_mask.size
        
        # If any disease color is significant, mark as disease region
        return (brown_ratio > 0.05) or (yellow_ratio > 0.05) or (white_ratio > 0.05)
    
    def process_regions(self, image: np.ndarray, 
                       region_params: List[List[Dict]],
                       region_h: int, region_w: int) -> np.ndarray:
        """
        Process each region with its specific parameters
        
        Args:
            image: Input image
            region_params: Parameters for each region
            region_h: Height of each region
            region_w: Width of each region
            
        Returns:
            Processed image with per-region enhancements
        """
        result = np.zeros_like(image)
        
        for r, row_params in enumerate(region_params):
            for c, params in enumerate(row_params):
                # Extract region
                y_start, y_end, x_start, x_end = params['bounds']
                region = image[y_start:y_end, x_start:x_end].copy()
                
                # Process region based on parameters
                processed = self.process_single_region(region, params)
                
                # Place processed region in result
                result[y_start:y_end, x_start:x_end] = processed
        
        return result
    
    def process_single_region(self, region: np.ndarray, params: Dict) -> np.ndarray:
        """
        Process a single region with specific parameters
        
        Args:
            region: Image region
            params: Processing parameters
            
        Returns:
            Processed region
        """
        # Convert to LAB for processing
        lab = cv2.cvtColor(region, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]
        
        # Apply gamma correction
        if params['gamma'] != 1.0:
            l_channel = self.apply_gamma(l_channel, params['gamma'])
        
        # Apply CLAHE with region-specific parameters
        clahe = cv2.createCLAHE(clipLimit=params['clahe_clip'], tileGridSize=(4, 4))
        l_enhanced = clahe.apply(l_channel)
        
        # Blend based on enhancement strength
        l_final = (l_channel * (1 - params['enhance_strength']) + 
                  l_enhanced * params['enhance_strength'])
        
        # Update LAB image
        lab[:, :, 0] = np.clip(l_final, 0, 255).astype(np.uint8)
        
        # Convert back to RGB
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Additional processing for specific region types
        if params['type'] == 'overexposed':
            result = self.reduce_highlights(result)
        elif params['type'] == 'underexposed':
            result = self.boost_shadows(result)
        
        return result
    
    def apply_gamma(self, channel: np.ndarray, gamma: float) -> np.ndarray:
        """
        Apply gamma correction to a channel
        
        Args:
            channel: Single channel image
            gamma: Gamma value
            
        Returns:
            Gamma corrected channel
        """
        # Normalize to [0, 1]
        normalized = channel.astype(np.float32) / 255.0
        
        # Apply gamma
        corrected = np.power(normalized, 1.0 / gamma)
        
        # Convert back to [0, 255]
        return (corrected * 255).astype(np.uint8)
    
    def reduce_highlights(self, region: np.ndarray) -> np.ndarray:
        """
        Reduce highlights in overexposed regions
        
        Args:
            region: Overexposed region
            
        Returns:
            Region with reduced highlights
        """
        # Simple highlight compression
        region_float = region.astype(np.float32)
        
        # Compress values above 200
        mask = region_float > 200
        region_float[mask] = 200 + (region_float[mask] - 200) * 0.3
        
        return np.clip(region_float, 0, 255).astype(np.uint8)
    
    def boost_shadows(self, region: np.ndarray) -> np.ndarray:
        """
        Boost detail in shadow regions
        
        Args:
            region: Underexposed region
            
        Returns:
            Region with boosted shadows
        """
        # Shadow lifting using logarithmic curve
        region_float = region.astype(np.float32) / 255.0
        
        # Apply log curve to shadows
        shadow_mask = region_float < 0.3
        region_float[shadow_mask] = np.log1p(region_float[shadow_mask] * 2) / np.log(3)
        
        return (np.clip(region_float, 0, 1) * 255).astype(np.uint8)
    
    def blend_regions(self, processed: np.ndarray, 
                     region_h: int, region_w: int) -> np.ndarray:
        """
        Blend regions with smooth transitions to avoid visible boundaries
        
        Args:
            processed: Image with processed regions
            region_h: Height of each region
            region_w: Width of each region
            
        Returns:
            Smoothly blended result
        """
        # Create weight map for smooth blending
        h, w = processed.shape[:2]
        rows, cols = self.grid_size
        
        # Generate smooth transition weights
        weight_map = self.create_smooth_weight_map(h, w, region_h, region_w)
        
        # Apply bilateral filter for edge-aware smoothing
        # This preserves disease pattern edges while smoothing region boundaries
        result = cv2.bilateralFilter(processed, 9, 75, 75)
        
        # Additional smoothing at region boundaries
        for r in range(1, rows):
            y = r * region_h
            if y < h:
                # Smooth horizontal boundary
                boundary_region = result[max(0, y-10):min(h, y+10), :]
                smoothed = cv2.GaussianBlur(boundary_region, (1, 21), 0)
                result[max(0, y-10):min(h, y+10), :] = smoothed
        
        for c in range(1, cols):
            x = c * region_w
            if x < w:
                # Smooth vertical boundary
                boundary_region = result[:, max(0, x-10):min(w, x+10)]
                smoothed = cv2.GaussianBlur(boundary_region, (21, 1), 0)
                result[:, max(0, x-10):min(w, x+10)] = smoothed
        
        return result
    
    def create_smooth_weight_map(self, h: int, w: int,
                                 region_h: int, region_w: int) -> np.ndarray:
        """
        Create smooth weight map for blending regions
        
        Args:
            h, w: Image dimensions
            region_h, region_w: Region dimensions
            
        Returns:
            Weight map for smooth blending
        """
        weight_map = np.ones((h, w), dtype=np.float32)
        rows, cols = self.grid_size
        
        # Create gradients at boundaries
        for r in range(rows):
            for c in range(cols):
                y_start = r * region_h
                y_end = min((r + 1) * region_h, h)
                x_start = c * region_w
                x_end = min((c + 1) * region_w, w)
                
                # Create gradient weights near boundaries
                border_size = 20
                
                # Top border
                if r > 0 and y_start + border_size < h:
                    for i in range(border_size):
                        weight = i / border_size
                        weight_map[y_start + i, x_start:x_end] *= weight
                
                # Left border
                if c > 0 and x_start + border_size < w:
                    for i in range(border_size):
                        weight = i / border_size
                        weight_map[y_start:y_end, x_start + i] *= weight
        
        # Smooth the weight map
        weight_map = cv2.GaussianBlur(weight_map, (31, 31), 0)
        
        return weight_map