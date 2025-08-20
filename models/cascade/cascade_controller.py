"""
Three-Tier Model Cascade Controller
Orchestrates intelligent routing between model tiers based on confidence
Tier 1: EfficientFormer-L7 (7ms) - Easy cases
Tier 2: EfficientNet-B4 (600-800ms) - Moderate cases  
Tier 3: CNN-ViT Ensemble (1.2-1.5s) - Hard cases
"""

import numpy as np
import torch
from typing import Dict, Optional, Tuple, List
import time
import logging

logger = logging.getLogger(__name__)


class ModelCascadeController:
    """
    Intelligent cascade controller for three-tier inference
    Routes samples to appropriate tier based on confidence and complexity
    """
    
    def __init__(self, 
                 enable_tier2: bool = True,
                 enable_tier3: bool = True,
                 device: str = 'cpu'):
        """
        Initialize cascade controller
        
        Args:
            enable_tier2: Enable Tier 2 model
            enable_tier3: Enable Tier 3 model
            device: Inference device
        """
        self.device = device
        self.enable_tier2 = enable_tier2
        self.enable_tier3 = enable_tier3
        
        # Initialize Tier 1 (always enabled)
        from models.architectures.efficientformer import EfficientFormerTier1
        self.tier1 = EfficientFormerTier1(device=device)
        logger.info("Tier 1 (EfficientFormer-L7) initialized")
        
        # Initialize Tier 2 if enabled
        self.tier2 = None
        if enable_tier2:
            try:
                # Will implement EfficientNet-B4 next
                logger.info("Tier 2 placeholder - EfficientNet-B4 to be implemented")
            except Exception as e:
                logger.warning(f"Tier 2 initialization failed: {e}")
                self.enable_tier2 = False
        
        # Initialize Tier 3 if enabled
        self.tier3 = None
        if enable_tier3:
            try:
                # Will implement CNN-ViT Ensemble later
                logger.info("Tier 3 placeholder - CNN-ViT Ensemble to be implemented")
            except Exception as e:
                logger.warning(f"Tier 3 initialization failed: {e}")
                self.enable_tier3 = False
        
        # Cascade thresholds
        self.tier1_threshold = 0.85  # Below this, escalate to Tier 2
        self.tier2_threshold = 0.80  # Below this, escalate to Tier 3
        self.unknown_threshold = 0.70  # Below this, classify as Unknown
        
        # Complexity indicators that trigger higher tiers
        self.complexity_indicators = {
            'multiple_diseases': 0.3,  # Multiple high probabilities
            'ambiguous_pattern': 0.4,  # Similar probabilities for different diseases
            'low_contrast': 0.5,       # Poor image quality
            'edge_case': 0.6          # Unusual patterns
        }
        
        # Statistics tracking
        self.stats = {
            'total_inferences': 0,
            'tier1_handled': 0,
            'tier2_handled': 0,
            'tier3_handled': 0,
            'unknown_classified': 0,
            'avg_inference_time': 0,
            'avg_confidence': 0
        }
    
    def infer(self, image: np.ndarray, 
              preprocessed_data: Optional[Dict] = None,
              force_tier: Optional[int] = None) -> Dict:
        """
        Run cascade inference
        
        Args:
            image: RGB image array
            preprocessed_data: Optional preprocessed features
            force_tier: Force specific tier (for testing)
            
        Returns:
            Inference results with cascade metadata
        """
        start_time = time.time()
        
        # Track cascade path
        cascade_path = []
        
        # Assess image complexity if preprocessed data available
        complexity_score = 0.0
        if preprocessed_data:
            complexity_score = self.assess_complexity(preprocessed_data)
        
        # Determine starting tier based on complexity
        if force_tier:
            starting_tier = force_tier
        elif complexity_score > 0.7:
            starting_tier = 3 if self.enable_tier3 else 2
        elif complexity_score > 0.4:
            starting_tier = 2 if self.enable_tier2 else 1
        else:
            starting_tier = 1
        
        # TIER 1 INFERENCE
        if starting_tier <= 1:
            tier1_result = self.tier1.infer(image)
            cascade_path.append('tier1')
            
            # Check if Tier 1 is confident enough
            if tier1_result['confidence'] >= self.tier1_threshold or not self.enable_tier2:
                # Tier 1 handles it
                self.stats['tier1_handled'] += 1
                
                return self.finalize_result(
                    tier1_result, 
                    cascade_path,
                    start_time,
                    complexity_score
                )
            
            # Need to escalate to Tier 2
            logger.debug(f"Escalating to Tier 2: confidence {tier1_result['confidence']:.2f}")
        
        # TIER 2 INFERENCE
        if self.enable_tier2 and (starting_tier <= 2 or 
                                  (starting_tier == 1 and tier1_result.get('should_escalate', False))):
            
            # Placeholder for Tier 2 (will implement EfficientNet-B4)
            tier2_result = self.mock_tier2_inference(image)
            cascade_path.append('tier2')
            
            # Check if Tier 2 is confident enough
            if tier2_result['confidence'] >= self.tier2_threshold or not self.enable_tier3:
                # Tier 2 handles it
                self.stats['tier2_handled'] += 1
                
                return self.finalize_result(
                    tier2_result,
                    cascade_path,
                    start_time,
                    complexity_score
                )
            
            # Need to escalate to Tier 3
            logger.debug(f"Escalating to Tier 3: confidence {tier2_result['confidence']:.2f}")
        
        # TIER 3 INFERENCE
        if self.enable_tier3:
            # Placeholder for Tier 3 (will implement CNN-ViT Ensemble)
            tier3_result = self.mock_tier3_inference(image)
            cascade_path.append('tier3')
            
            self.stats['tier3_handled'] += 1
            
            return self.finalize_result(
                tier3_result,
                cascade_path,
                start_time,
                complexity_score
            )
        
        # Fallback - return best available result
        if 'tier2_result' in locals():
            return self.finalize_result(tier2_result, cascade_path, start_time, complexity_score)
        elif 'tier1_result' in locals():
            return self.finalize_result(tier1_result, cascade_path, start_time, complexity_score)
        else:
            # Should never reach here
            return self.create_unknown_result(cascade_path, start_time)
    
    def assess_complexity(self, preprocessed_data: Dict) -> float:
        """
        Assess image complexity to determine starting tier
        
        Args:
            preprocessed_data: Preprocessed features
            
        Returns:
            Complexity score (0-1)
        """
        complexity = 0.0
        
        # Check for multiple disease indicators
        if 'disease_regions' in preprocessed_data:
            disease_count = preprocessed_data['disease_regions'].get('count', 0)
            if disease_count > 2:
                complexity += self.complexity_indicators['multiple_diseases']
        
        # Check image quality
        if 'quality_metrics' in preprocessed_data:
            quality = preprocessed_data['quality_metrics']
            
            # Low contrast
            if quality.get('contrast', 1.0) < 0.3:
                complexity += self.complexity_indicators['low_contrast']
            
            # Blur/noise
            if quality.get('sharpness', 1.0) < 0.4:
                complexity += self.complexity_indicators['edge_case']
        
        # Check segmentation confidence
        if 'segmentation_quality' in preprocessed_data:
            seg_quality = preprocessed_data['segmentation_quality']
            if seg_quality < 0.5:
                complexity += self.complexity_indicators['ambiguous_pattern']
        
        return min(complexity, 1.0)
    
    def finalize_result(self, result: Dict, cascade_path: List[str],
                       start_time: float, complexity_score: float) -> Dict:
        """
        Finalize and augment inference result
        
        Args:
            result: Raw inference result
            cascade_path: Path through cascade
            start_time: Inference start time
            complexity_score: Image complexity
            
        Returns:
            Final result dictionary
        """
        # Calculate total time
        total_time = (time.time() - start_time) * 1000
        
        # Update statistics
        self.stats['total_inferences'] += 1
        self.stats['avg_inference_time'] = (
            (self.stats['avg_inference_time'] * (self.stats['total_inferences'] - 1) + 
             total_time) / self.stats['total_inferences']
        )
        self.stats['avg_confidence'] = (
            (self.stats['avg_confidence'] * (self.stats['total_inferences'] - 1) + 
             result.get('confidence', 0)) / self.stats['total_inferences']
        )
        
        # Check for Unknown classification
        if result.get('confidence', 0) < self.unknown_threshold:
            result['class'] = 'Unknown'
            self.stats['unknown_classified'] += 1
        
        # Augment result
        result.update({
            'cascade_path': cascade_path,
            'total_inference_time_ms': total_time,
            'complexity_score': complexity_score,
            'cascade_stats': self.get_stats()
        })
        
        return result
    
    def create_unknown_result(self, cascade_path: List[str],
                             start_time: float) -> Dict:
        """Create result for unknown/failed classification"""
        return {
            'tier': 0,
            'class': 'Unknown',
            'confidence': 0.0,
            'probability': 0.0,
            'all_probabilities': [1/7] * 7,  # Uniform distribution
            'cascade_path': cascade_path,
            'total_inference_time_ms': (time.time() - start_time) * 1000,
            'error': 'Classification failed'
        }
    
    def mock_tier2_inference(self, image: np.ndarray) -> Dict:
        """Placeholder for Tier 2 inference"""
        # Simulate Tier 2 processing time
        time.sleep(0.6)  # 600ms
        
        return {
            'tier': 2,
            'class': 'Blight',
            'confidence': 0.92,
            'probability': 0.92,
            'all_probabilities': [0.02, 0.92, 0.02, 0.01, 0.01, 0.01, 0.01],
            'inference_time_ms': 600
        }
    
    def mock_tier3_inference(self, image: np.ndarray) -> Dict:
        """Placeholder for Tier 3 inference"""
        # Simulate Tier 3 processing time
        time.sleep(1.2)  # 1200ms
        
        return {
            'tier': 3,
            'class': 'Mosaic Virus',
            'confidence': 0.98,
            'probability': 0.98,
            'all_probabilities': [0.005, 0.005, 0.005, 0.98, 0.002, 0.002, 0.001],
            'inference_time_ms': 1200
        }
    
    def update_thresholds(self, tier1: float = None,
                         tier2: float = None,
                         unknown: float = None):
        """
        Update cascade thresholds
        
        Args:
            tier1: New Tier 1 threshold
            tier2: New Tier 2 threshold
            unknown: New Unknown threshold
        """
        if tier1 is not None:
            self.tier1_threshold = tier1
        if tier2 is not None:
            self.tier2_threshold = tier2
        if unknown is not None:
            self.unknown_threshold = unknown
        
        logger.info(f"Updated thresholds: T1={self.tier1_threshold}, "
                   f"T2={self.tier2_threshold}, Unknown={self.unknown_threshold}")
    
    def get_stats(self) -> Dict:
        """Get cascade statistics"""
        stats = self.stats.copy()
        
        # Calculate percentages
        total = stats['total_inferences']
        if total > 0:
            stats['tier1_percentage'] = (stats['tier1_handled'] / total) * 100
            stats['tier2_percentage'] = (stats['tier2_handled'] / total) * 100
            stats['tier3_percentage'] = (stats['tier3_handled'] / total) * 100
            stats['unknown_percentage'] = (stats['unknown_classified'] / total) * 100
        
        return stats
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            'total_inferences': 0,
            'tier1_handled': 0,
            'tier2_handled': 0,
            'tier3_handled': 0,
            'unknown_classified': 0,
            'avg_inference_time': 0,
            'avg_confidence': 0
        }
    
    def benchmark_cascade(self, test_images: List[np.ndarray]) -> Dict:
        """
        Benchmark cascade performance
        
        Args:
            test_images: List of test images
            
        Returns:
            Benchmark results
        """
        logger.info(f"Benchmarking cascade with {len(test_images)} images...")
        
        self.reset_stats()
        times = []
        
        for i, image in enumerate(test_images):
            start = time.time()
            result = self.infer(image)
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)
            
            if (i + 1) % 10 == 0:
                logger.debug(f"Processed {i+1}/{len(test_images)} images")
        
        stats = self.get_stats()
        
        return {
            'num_images': len(test_images),
            'mean_time_ms': np.mean(times),
            'std_time_ms': np.std(times),
            'min_time_ms': np.min(times),
            'max_time_ms': np.max(times),
            'tier_distribution': {
                'tier1': stats['tier1_percentage'],
                'tier2': stats['tier2_percentage'],
                'tier3': stats['tier3_percentage']
            },
            'unknown_rate': stats['unknown_percentage'],
            'avg_confidence': stats['avg_confidence']
        }
    
    def export_for_mobile(self) -> Dict:
        """Export models for Core ML conversion"""
        exports = {}
        
        # Export Tier 1
        if self.tier1:
            exports['tier1'] = self.tier1.model.export_for_mobile()
        
        # Export Tier 2 (when implemented)
        if self.tier2:
            pass  # exports['tier2'] = self.tier2.export_for_mobile()
        
        # Export Tier 3 (when implemented)
        if self.tier3:
            pass  # exports['tier3'] = self.tier3.export_for_mobile()
        
        return exports