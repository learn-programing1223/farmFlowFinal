"""
Disease Protection Layer for Segmentation
GUARANTEES that disease regions are never lost during segmentation
This is the most critical component - ensures 100% disease preservation
"""

import numpy as np
import cv2
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class DiseaseProtectionLayer:
    """
    Ensures disease regions are ALWAYS included in final segmentation
    Philosophy: Mathematical union guarantees preservation
    
    Key principle: Better to include extra background than lose any disease
    """
    
    def __init__(self, dilation_size: int = 5):
        """
        Initialize disease protection layer
        
        Args:
            dilation_size: Size of dilation kernel for disease boundaries
        """
        self.dilation_size = dilation_size
        
        # Create dilation kernel
        self.dilation_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (dilation_size, dilation_size)
        )
        
        # Stats
        self.stats = {
            'protections_applied': 0,
            'avg_disease_coverage': 0,
            'avg_expansion_ratio': 0
        }
    
    def protect(self, segmentation_mask: np.ndarray, 
                disease_mask: np.ndarray,
                validate: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Apply disease protection to segmentation
        
        Args:
            segmentation_mask: Initial segmentation
            disease_mask: Detected disease regions (MUST be preserved)
            validate: Whether to validate 100% preservation
            
        Returns:
            Tuple of (protected_mask, protection_info)
            
        Raises:
            AssertionError: If disease preservation fails validation
        """
        h, w = segmentation_mask.shape[:2]
        
        # Ensure masks are same size
        if disease_mask.shape[:2] != (h, w):
            disease_mask = cv2.resize(disease_mask, (w, h))
        
        # Protection info
        info = {
            'initial_coverage': 0,
            'final_coverage': 0,
            'disease_preserved': 0,
            'expansion_ratio': 0,
            'validation_passed': False
        }
        
        # Calculate initial coverage
        initial_plant_pixels = np.sum(segmentation_mask > 0)
        initial_disease_pixels = np.sum(disease_mask > 0)
        
        info['initial_coverage'] = initial_plant_pixels / (h * w)
        
        # Step 1: Dilate disease mask to include boundaries
        # Disease boundaries are critical - often where early symptoms appear
        disease_dilated = self.dilate_disease_regions(disease_mask)
        
        # Step 2: Mathematical union - GUARANTEES inclusion
        protected_mask = self.apply_protection(segmentation_mask, disease_dilated)
        
        # Step 3: Connect nearby disease regions to plant
        # If disease is detected near plant edge, connect them
        protected_mask = self.connect_disease_to_plant(protected_mask, disease_dilated)
        
        # Step 4: Smooth boundaries while preserving disease
        protected_mask = self.smooth_boundaries(protected_mask, disease_dilated)
        
        # Step 5: Final safety check - re-apply disease mask
        # This is redundant but ensures 100% preservation
        protected_mask = cv2.bitwise_or(protected_mask, disease_dilated)
        
        # Calculate final coverage
        final_plant_pixels = np.sum(protected_mask > 0)
        info['final_coverage'] = final_plant_pixels / (h * w)
        
        # Calculate expansion ratio
        if initial_plant_pixels > 0:
            info['expansion_ratio'] = final_plant_pixels / initial_plant_pixels
        else:
            info['expansion_ratio'] = float('inf')
        
        # Validate disease preservation
        if validate and initial_disease_pixels > 0:
            preserved = self.validate_preservation(protected_mask, disease_mask)
            info['disease_preserved'] = preserved
            info['validation_passed'] = (preserved >= 0.99)  # Allow 1% tolerance
            
            if not info['validation_passed']:
                logger.error(f"Disease preservation failed: {preserved:.1%}")
                # Force include all disease regions
                protected_mask = cv2.bitwise_or(protected_mask, disease_mask)
                # Re-validate
                preserved = self.validate_preservation(protected_mask, disease_mask)
                info['disease_preserved'] = preserved
                info['validation_passed'] = (preserved >= 0.99)
                
                assert info['validation_passed'], \
                    f"Critical: Disease preservation still failed after correction: {preserved:.1%}"
        
        # Update statistics
        self.update_stats(info)
        
        logger.debug(f"Disease protection applied: {info['disease_preserved']:.1%} preserved")
        
        return protected_mask, info
    
    def dilate_disease_regions(self, disease_mask: np.ndarray) -> np.ndarray:
        """
        Dilate disease regions to include boundaries
        Disease boundaries often contain important early symptoms
        
        Args:
            disease_mask: Binary disease mask
            
        Returns:
            Dilated disease mask
        """
        # Apply dilation
        dilated = cv2.dilate(disease_mask, self.dilation_kernel, iterations=1)
        
        # For very small disease spots, apply extra dilation
        # Small spots are often early disease indicators
        small_spots = self.find_small_disease_spots(disease_mask)
        if small_spots is not None:
            small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            small_dilated = cv2.dilate(small_spots, small_kernel, iterations=2)
            dilated = cv2.bitwise_or(dilated, small_dilated)
        
        return dilated
    
    def find_small_disease_spots(self, disease_mask: np.ndarray) -> Optional[np.ndarray]:
        """
        Find small disease spots that need extra protection
        
        Args:
            disease_mask: Binary disease mask
            
        Returns:
            Mask of small spots or None
        """
        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(disease_mask, connectivity=8)
        
        if num_labels <= 1:  # No disease regions
            return None
        
        # Find small components (area < 100 pixels)
        small_mask = np.zeros_like(disease_mask)
        
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area < 100:  # Small spot
                small_mask[labels == label] = 255
        
        if np.any(small_mask):
            return small_mask
        else:
            return None
    
    def apply_protection(self, plant_mask: np.ndarray, 
                        disease_mask: np.ndarray) -> np.ndarray:
        """
        Apply mathematical union to guarantee disease inclusion
        
        Args:
            plant_mask: Initial plant segmentation
            disease_mask: Disease regions to protect
            
        Returns:
            Protected mask with guaranteed disease inclusion
        """
        # Simple but powerful: mathematical union
        # This GUARANTEES all disease pixels are included
        protected = cv2.bitwise_or(plant_mask, disease_mask)
        
        return protected
    
    def connect_disease_to_plant(self, plant_mask: np.ndarray,
                                 disease_mask: np.ndarray) -> np.ndarray:
        """
        Connect isolated disease regions to main plant body
        Disease doesn't occur in isolation - it's always on plant material
        
        Args:
            plant_mask: Current plant mask
            disease_mask: Disease regions
            
        Returns:
            Mask with disease regions connected to plant
        """
        # Find disease regions not connected to plant
        # These might be disease on detached leaves or isolated spots
        
        # Find connected components in plant mask
        num_plant, plant_labels = cv2.connectedComponents(plant_mask)
        
        # Find largest plant component (main plant body)
        if num_plant > 1:
            # Count pixels in each component
            counts = np.bincount(plant_labels.flatten())
            # Ignore background (label 0)
            counts[0] = 0
            # Find largest component
            main_plant_label = np.argmax(counts)
            main_plant = (plant_labels == main_plant_label).astype(np.uint8) * 255
        else:
            main_plant = plant_mask
        
        # Check if disease regions are connected to main plant
        num_disease, disease_labels = cv2.connectedComponents(disease_mask)
        
        connected_mask = plant_mask.copy()
        
        for disease_label in range(1, num_disease):
            disease_region = (disease_labels == disease_label).astype(np.uint8) * 255
            
            # Check if this disease region overlaps with main plant
            overlap = cv2.bitwise_and(disease_region, main_plant)
            
            if np.sum(overlap) == 0:  # Not connected
                # Connect using morphological closing
                # This assumes disease is near plant (which it should be)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
                
                # Combine disease and plant
                combined = cv2.bitwise_or(disease_region, main_plant)
                
                # Close the gap
                closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
                
                # Add to connected mask
                connected_mask = cv2.bitwise_or(connected_mask, closed)
        
        return connected_mask
    
    def smooth_boundaries(self, mask: np.ndarray, 
                         disease_mask: np.ndarray) -> np.ndarray:
        """
        Smooth segmentation boundaries while preserving disease regions
        
        Args:
            mask: Current mask
            disease_mask: Disease regions to preserve
            
        Returns:
            Smoothed mask with disease preserved
        """
        # Smooth using morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Open to remove small protrusions
        smoothed = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Close to fill small gaps
        smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_CLOSE, kernel)
        
        # CRITICAL: Re-apply disease mask after smoothing
        # Smoothing might remove disease edges
        smoothed = cv2.bitwise_or(smoothed, disease_mask)
        
        return smoothed
    
    def validate_preservation(self, final_mask: np.ndarray,
                            original_disease: np.ndarray) -> float:
        """
        Validate that disease regions are preserved
        
        Args:
            final_mask: Final segmentation mask
            original_disease: Original disease mask
            
        Returns:
            Preservation ratio (1.0 = 100% preserved)
        """
        if np.sum(original_disease) == 0:
            # No disease to preserve
            return 1.0
        
        # Check how much of original disease is in final mask
        preserved = cv2.bitwise_and(final_mask, original_disease)
        
        preservation_ratio = np.sum(preserved > 0) / np.sum(original_disease > 0)
        
        return preservation_ratio
    
    def validate_coverage(self, mask: np.ndarray, 
                         disease_mask: np.ndarray) -> Dict:
        """
        Comprehensive validation of disease coverage
        
        Args:
            mask: Segmentation mask
            disease_mask: Disease mask
            
        Returns:
            Validation results
        """
        results = {
            'disease_pixels': np.sum(disease_mask > 0),
            'preserved_pixels': 0,
            'preservation_ratio': 0.0,
            'missed_regions': 0,
            'passed': False
        }
        
        if results['disease_pixels'] == 0:
            results['passed'] = True
            results['preservation_ratio'] = 1.0
            return results
        
        # Check pixel-wise preservation
        preserved = cv2.bitwise_and(mask, disease_mask)
        results['preserved_pixels'] = np.sum(preserved > 0)
        results['preservation_ratio'] = results['preserved_pixels'] / results['disease_pixels']
        
        # Check connected components
        num_disease, disease_labels = cv2.connectedComponents(disease_mask)
        
        missed = 0
        for label in range(1, num_disease):
            region = (disease_labels == label).astype(np.uint8) * 255
            overlap = cv2.bitwise_and(region, mask)
            
            if np.sum(overlap) == 0:
                missed += 1
        
        results['missed_regions'] = missed
        results['passed'] = (results['preservation_ratio'] >= 0.99 and missed == 0)
        
        return results
    
    def update_stats(self, info: Dict):
        """Update protection statistics"""
        n = self.stats['protections_applied'] + 1
        self.stats['protections_applied'] = n
        
        # Update averages
        if 'disease_preserved' in info:
            self.stats['avg_disease_coverage'] = \
                (self.stats['avg_disease_coverage'] * (n-1) + info['disease_preserved']) / n
        
        if 'expansion_ratio' in info and info['expansion_ratio'] != float('inf'):
            self.stats['avg_expansion_ratio'] = \
                (self.stats['avg_expansion_ratio'] * (n-1) + info['expansion_ratio']) / n
    
    def get_stats(self) -> Dict:
        """Get protection statistics"""
        return self.stats.copy()