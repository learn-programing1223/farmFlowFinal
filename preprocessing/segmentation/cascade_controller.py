"""
Segmentation Cascade Controller
Orchestrates the disease-preserving segmentation pipeline
Implements the two-path cascade with automatic path selection
"""

import numpy as np
import cv2
from typing import Dict, Tuple, Optional, Union
from pathlib import Path
import logging
import time

from .disease_detector import DiseaseRegionDetector
from .rgb_segmentation import RGBSegmentation
from .disease_protection import DiseaseProtectionLayer

logger = logging.getLogger(__name__)


class SegmentationCascade:
    """
    Main segmentation controller implementing disease-first philosophy
    
    Pipeline:
    1. Detect disease regions (ALWAYS)
    2. Analyze background complexity
    3. Choose segmentation path (RGB or DeepLab)
    4. Apply disease protection (ALWAYS)
    5. Fallback if needed
    """
    
    def __init__(self, 
                 enable_deeplab: bool = False,
                 device: str = 'cpu'):
        """
        Initialize segmentation cascade
        
        Args:
            enable_deeplab: Whether to use DeepLab for complex backgrounds
            device: Device for deep learning models ('cpu' or 'cuda')
        """
        # Initialize components
        self.disease_detector = DiseaseRegionDetector(sensitivity=0.8)
        self.rgb_segmenter = RGBSegmentation()
        self.disease_protector = DiseaseProtectionLayer(dilation_size=5)
        
        # DeepLab is optional (can work with RGB only)
        self.enable_deeplab = enable_deeplab
        self.deeplab = None
        
        if enable_deeplab:
            try:
                from .deeplab_segmentation import DeepLabSegmentation
                self.deeplab = DeepLabSegmentation(device=device)
                logger.info("DeepLab segmentation enabled")
            except Exception as e:
                logger.warning(f"Could not load DeepLab: {e}. Using RGB only.")
                self.enable_deeplab = False
        
        # Complexity threshold for path selection
        self.complexity_threshold = 0.3
        
        # Quality threshold for fallback
        self.quality_threshold = 0.5
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'rgb_path_used': 0,
            'deeplab_path_used': 0,
            'fallback_used': 0,
            'avg_time_ms': 0,
            'avg_disease_coverage': 0
        }
    
    def segment(self, 
                image: np.ndarray,
                vegetation_indices: Optional[Dict] = None,
                use_fallback: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Main segmentation method with disease preservation
        
        Args:
            image: RGB image to segment
            vegetation_indices: Pre-computed indices from pipeline
            use_fallback: Whether to use fallback strategies
            
        Returns:
            Tuple of (segmentation_mask, segmentation_info)
        """
        start_time = time.time()
        h, w = image.shape[:2]
        
        # Segmentation info
        info = {
            'disease_detected': False,
            'disease_coverage': 0,
            'path_used': 'none',
            'quality_score': 0,
            'fallback_used': False,
            'time_ms': 0,
            'preservation_validated': False
        }
        
        try:
            # Step 1: ALWAYS detect disease regions first
            logger.debug("Detecting disease regions...")
            disease_mask, disease_info = self.disease_detector.detect(image)
            
            info['disease_detected'] = disease_info['total_coverage'] > 0
            info['disease_coverage'] = disease_info['total_coverage']
            
            if info['disease_detected']:
                logger.info(f"Disease detected: {info['disease_coverage']:.1%} coverage")
            
            # Step 2: Analyze background complexity
            complexity = self.analyze_background_complexity(image)
            logger.debug(f"Background complexity: {complexity:.2f}")
            
            # Step 3: Choose segmentation path
            if complexity < self.complexity_threshold or not self.enable_deeplab:
                # Simple background or DeepLab not available - use RGB
                logger.debug("Using RGB segmentation path")
                plant_mask, seg_info = self.rgb_segmenter.segment(
                    image, 
                    disease_mask=disease_mask,
                    vegetation_indices=vegetation_indices
                )
                info['path_used'] = 'rgb'
                self.stats['rgb_path_used'] += 1
                
            else:
                # Complex background - use DeepLab if available
                if self.deeplab is not None:
                    logger.debug("Using DeepLab segmentation path")
                    plant_mask, seg_info = self.deeplab.segment(image, disease_mask)
                    info['path_used'] = 'deeplab'
                    self.stats['deeplab_path_used'] += 1
                else:
                    # Fallback to RGB if DeepLab not available
                    logger.debug("DeepLab not available, using RGB")
                    plant_mask, seg_info = self.rgb_segmenter.segment(
                        image,
                        disease_mask=disease_mask,
                        vegetation_indices=vegetation_indices
                    )
                    info['path_used'] = 'rgb'
                    self.stats['rgb_path_used'] += 1
            
            # Step 4: ALWAYS apply disease protection
            logger.debug("Applying disease protection...")
            final_mask, protection_info = self.disease_protector.protect(
                plant_mask,
                disease_mask,
                validate=True
            )
            
            info['preservation_validated'] = protection_info['validation_passed']
            
            # Step 5: Evaluate quality and use fallback if needed
            quality = self.evaluate_segmentation_quality(final_mask, image)
            info['quality_score'] = quality
            
            if use_fallback and quality < self.quality_threshold:
                logger.warning(f"Low quality score: {quality:.2f}, using fallback")
                final_mask = self.apply_fallback(image, disease_mask)
                info['fallback_used'] = True
                self.stats['fallback_used'] += 1
            
            # Ensure final mask is correct type
            final_mask = final_mask.astype(np.uint8)
            if np.max(final_mask) == 1:
                final_mask = final_mask * 255
            
        except Exception as e:
            logger.error(f"Segmentation failed: {e}")
            # Emergency fallback - return center crop
            final_mask = self.center_crop_fallback(image)
            info['fallback_used'] = True
            info['path_used'] = 'emergency_fallback'
        
        # Calculate timing
        elapsed_ms = (time.time() - start_time) * 1000
        info['time_ms'] = elapsed_ms
        
        # Update statistics
        self.update_stats(info)
        
        logger.info(f"Segmentation complete: {info['path_used']} path, "
                   f"{elapsed_ms:.1f}ms, quality: {info['quality_score']:.2f}")
        
        return final_mask, info
    
    def analyze_background_complexity(self, image: np.ndarray) -> float:
        """
        Analyze background complexity to choose segmentation path
        
        Args:
            image: RGB image
            
        Returns:
            Complexity score (0-1, higher = more complex)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 1. Edge density (more edges = more complex)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # 2. Color variance
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hue_std = np.std(hsv[:, :, 0])
        sat_std = np.std(hsv[:, :, 1])
        color_variance = (hue_std + sat_std) / 360.0  # Normalize
        
        # 3. Texture complexity (using local variance)
        kernel_size = 5
        mean = cv2.blur(gray, (kernel_size, kernel_size))
        sq_mean = cv2.blur(gray**2, (kernel_size, kernel_size))
        variance = sq_mean - mean**2
        texture_complexity = np.mean(variance) / 1000.0  # Normalize
        
        # 4. Number of distinct regions
        # Use simple thresholding to estimate regions
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        num_labels, _ = cv2.connectedComponents(binary)
        region_complexity = min(num_labels / 20.0, 1.0)  # Normalize
        
        # Combine metrics
        complexity = (
            edge_density * 0.3 +
            color_variance * 0.2 +
            texture_complexity * 0.3 +
            region_complexity * 0.2
        )
        
        return min(complexity, 1.0)
    
    def evaluate_segmentation_quality(self, mask: np.ndarray, 
                                     image: np.ndarray) -> float:
        """
        Evaluate the quality of segmentation
        
        Args:
            mask: Segmentation mask
            image: Original image
            
        Returns:
            Quality score (0-1)
        """
        h, w = mask.shape[:2]
        
        # 1. Coverage check
        coverage = np.sum(mask > 0) / (h * w)
        
        if coverage < 0.02:  # Too little
            return 0.1
        elif coverage > 0.98:  # Too much (likely failed)
            return 0.2
        
        # 2. Connectivity check
        num_labels, labels = cv2.connectedComponents(mask)
        
        # Find largest component
        if num_labels > 1:
            counts = np.bincount(labels.flatten())
            counts[0] = 0  # Ignore background
            largest_size = np.max(counts)
            total_foreground = np.sum(counts)
            
            # Ratio of largest component to total
            if total_foreground > 0:
                largest_ratio = largest_size / total_foreground
            else:
                largest_ratio = 0
        else:
            largest_ratio = 1.0
        
        # 3. Shape regularity
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Calculate solidity
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            contour_area = cv2.contourArea(largest_contour)
            
            if hull_area > 0:
                solidity = contour_area / hull_area
            else:
                solidity = 0
            
            # Calculate circularity
            perimeter = cv2.arcLength(largest_contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * contour_area / (perimeter ** 2)
            else:
                circularity = 0
        else:
            solidity = 0
            circularity = 0
        
        # 4. Edge alignment with image
        # Good segmentation should align with image edges
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image_edges = cv2.Canny(gray, 50, 150)
        
        # Dilate mask boundary
        mask_boundary = cv2.Canny(mask, 128, 255)
        kernel = np.ones((3, 3), np.uint8)
        mask_boundary_dilated = cv2.dilate(mask_boundary, kernel, iterations=2)
        
        # Check overlap
        overlap = cv2.bitwise_and(image_edges, mask_boundary_dilated)
        if np.sum(mask_boundary) > 0:
            edge_alignment = np.sum(overlap) / np.sum(mask_boundary)
        else:
            edge_alignment = 0
        
        # Combine scores
        quality = (
            min(coverage * 3, 1.0) * 0.2 +  # Coverage (optimal around 0.33)
            largest_ratio * 0.2 +             # Connectivity
            solidity * 0.2 +                  # Shape regularity
            circularity * 0.1 +               # Circularity
            edge_alignment * 0.3               # Edge alignment
        )
        
        return quality
    
    def apply_fallback(self, image: np.ndarray, 
                      disease_mask: np.ndarray) -> np.ndarray:
        """
        Apply fallback segmentation strategy
        
        Args:
            image: RGB image
            disease_mask: Disease regions to preserve
            
        Returns:
            Fallback segmentation mask
        """
        h, w = image.shape[:2]
        
        # Strategy 1: Try GrabCut with disease regions as foreground hints
        try:
            mask = self.grabcut_with_hints(image, disease_mask)
            quality = self.evaluate_segmentation_quality(mask, image)
            
            if quality > 0.4:
                logger.info("Fallback: GrabCut successful")
                return mask
        except Exception as e:
            logger.debug(f"GrabCut fallback failed: {e}")
        
        # Strategy 2: Center crop (most plants are centered)
        mask = self.center_crop_fallback(image)
        
        # Always include disease regions
        if disease_mask is not None:
            mask = cv2.bitwise_or(mask, disease_mask)
        
        logger.info("Fallback: Using center crop")
        return mask
    
    def grabcut_with_hints(self, image: np.ndarray,
                          disease_mask: np.ndarray) -> np.ndarray:
        """
        Use GrabCut with disease regions as foreground hints
        
        Args:
            image: RGB image
            disease_mask: Disease regions (definite foreground)
            
        Returns:
            Segmentation mask
        """
        h, w = image.shape[:2]
        
        # Initialize mask
        mask = np.zeros((h, w), np.uint8)
        
        # Set disease regions as definite foreground
        mask[disease_mask > 0] = cv2.GC_FGD
        
        # Set center region as probable foreground
        center_x, center_y = w // 2, h // 2
        radius_x, radius_y = w // 3, h // 3
        
        cv2.ellipse(mask, (center_x, center_y), (radius_x, radius_y),
                   0, 0, 360, cv2.GC_PR_FGD, -1)
        
        # Set borders as definite background
        border_size = 10
        mask[:border_size, :] = cv2.GC_BGD
        mask[-border_size:, :] = cv2.GC_BGD
        mask[:, :border_size] = cv2.GC_BGD
        mask[:, -border_size:] = cv2.GC_BGD
        
        # Apply GrabCut
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        cv2.grabCut(image, mask, None, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)
        
        # Extract foreground
        result_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0)
        result_mask = result_mask.astype(np.uint8)
        
        # Ensure disease regions are included
        result_mask = cv2.bitwise_or(result_mask, disease_mask)
        
        return result_mask
    
    def center_crop_fallback(self, image: np.ndarray) -> np.ndarray:
        """
        Simple center crop fallback - most plants are centered
        
        Args:
            image: RGB image
            
        Returns:
            Center crop mask
        """
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Create elliptical center region (70% of image)
        center_x, center_y = w // 2, h // 2
        radius_x = int(w * 0.35)
        radius_y = int(h * 0.35)
        
        cv2.ellipse(mask, (center_x, center_y), (radius_x, radius_y),
                   0, 0, 360, 255, -1)
        
        return mask
    
    def update_stats(self, info: Dict):
        """Update cascade statistics"""
        n = self.stats['total_processed'] + 1
        self.stats['total_processed'] = n
        
        # Update averages
        if 'time_ms' in info:
            self.stats['avg_time_ms'] = \
                (self.stats['avg_time_ms'] * (n-1) + info['time_ms']) / n
        
        if 'disease_coverage' in info:
            self.stats['avg_disease_coverage'] = \
                (self.stats['avg_disease_coverage'] * (n-1) + info['disease_coverage']) / n
    
    def get_stats(self) -> Dict:
        """Get cascade statistics"""
        return self.stats.copy()