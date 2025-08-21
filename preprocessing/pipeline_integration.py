"""
Pipeline Integration for LASSR with other preprocessing components
Connects LASSR super-resolution with the full preprocessing pipeline
"""

import numpy as np
import cv2
import time
from typing import Dict, Optional, Tuple, List, Union
from pathlib import Path
import logging

from .lassr import LASSRProcessor
from .lassr_utils import detect_disease_regions, apply_clahe, correct_exposure
from .illumination.retinex_illumination import RetinexIllumination
from .illumination.extreme_conditions import ExtremeLightingHandler
from .illumination.local_adaptive import LocalAdaptiveProcessor
from .illumination.shadow_highlight import ShadowHighlightRecovery
from .illumination.edge_cases import EdgeCaseHandler
from .heic_handler import HEICProcessor


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """
    Complete preprocessing pipeline for FarmFlow
    Integrates all preprocessing components in the correct order
    
    Pipeline order (from research):
    1. LASSR super-resolution (200-400ms) - 21% accuracy gain
    2. Multi-resolution processing (384x384 or 512x512)
    3. Illumination normalization (150-200ms) - 8-12% gain
    4. Segmentation (300-500ms) - 30-40% gain
    5. Color space fusion (50-100ms) - 5-8% gain
    6. Vegetation indices (20-50ms) - 3-5% gain
    
    Total: 400-600ms target, 25-35% accuracy improvement
    """
    
    def __init__(self,
                 enable_lassr: bool = True,
                 enable_segmentation: bool = True,
                 enable_illumination: bool = True,
                 enable_advanced_illumination: bool = True,
                 device: str = 'auto'):
        """
        Initialize preprocessing pipeline
        
        Args:
            enable_lassr: Whether to use LASSR super-resolution
            enable_segmentation: Whether to use U-Net segmentation
            enable_illumination: Whether to use illumination normalization
            enable_advanced_illumination: Whether to use Day 3 advanced components
            device: 'cuda', 'cpu', or 'auto'
        """
        self.enable_lassr = enable_lassr
        self.enable_segmentation = enable_segmentation
        self.enable_illumination = enable_illumination
        self.enable_advanced_illumination = enable_advanced_illumination
        
        # Initialize LASSR if enabled
        if enable_lassr:
            self.lassr = LASSRProcessor(
                model_type='standard',
                device=device if device != 'auto' else None,
                optimize_mobile=True
            )
        else:
            self.lassr = None
        
        # Initialize illumination normalization
        if enable_illumination:
            self.illumination = RetinexIllumination(clahe_clip_limit=3.0)
        else:
            self.illumination = None
        
        # Initialize Day 3 advanced components
        if enable_advanced_illumination:
            self.extreme_handler = ExtremeLightingHandler()
            self.local_adaptive = LocalAdaptiveProcessor()
            self.shadow_highlight = ShadowHighlightRecovery()
            self.edge_handler = EdgeCaseHandler()
            self.heic_processor = HEICProcessor()
        else:
            self.extreme_handler = None
            self.local_adaptive = None
            self.shadow_highlight = None
            self.edge_handler = None
            self.heic_processor = None
        
        # Initialize segmentation if enabled
        if enable_segmentation:
            from .segmentation.cascade_controller import SegmentationCascade
            self.segmentation = SegmentationCascade()
        else:
            self.segmentation = None
        
        # Pipeline statistics
        self.stats = {
            'total_processed': 0,
            'avg_time_ms': 0,
            'component_times': {
                'lassr': 0,
                'segmentation': 0,
                'illumination': 0,
                'color_fusion': 0,
                'indices': 0
            }
        }
    
    def process(self,
                image: Union[np.ndarray, str, Path],
                target_size: Tuple[int, int] = (384, 384),
                return_intermediate: bool = False) -> Union[np.ndarray, Dict]:
        """
        Process image through complete pipeline
        
        Args:
            image: Input image or path
            target_size: Target resolution after processing
            return_intermediate: Return intermediate results for debugging
        
        Returns:
            Processed image or dictionary with intermediate results
        """
        start_time = time.time()
        timings = {}
        intermediates = {}
        
        # Load image if path
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            if image is None:
                raise ValueError(f"Could not load image from {image}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        original = image.copy()
        intermediates['original'] = original
        
        # Step 1: LASSR Super-Resolution (200-400ms)
        if self.enable_lassr and self.lassr is not None:
            t0 = time.time()
            image = self.lassr.process(image, preserve_disease=True)
            timings['lassr'] = (time.time() - t0) * 1000
            intermediates['lassr'] = image.copy()
            logger.info(f"LASSR completed in {timings['lassr']:.1f}ms")
        
        # Step 2: Multi-resolution processing
        t0 = time.time()
        image = self._apply_multi_resolution(image, target_size)
        timings['multi_resolution'] = (time.time() - t0) * 1000
        intermediates['multi_resolution'] = image.copy()
        
        # Step 3: Illumination Normalization (150-200ms)
        if self.enable_illumination:
            t0 = time.time()
            image = self._apply_illumination_normalization(image)
            timings['illumination'] = (time.time() - t0) * 1000
            intermediates['illumination'] = image.copy()
            logger.info(f"Illumination normalization in {timings['illumination']:.1f}ms")
        
        # Step 3b: Advanced Illumination (Day 3 enhancements) - if needed
        if self.enable_advanced_illumination and self.extreme_handler:
            # Check if extreme conditions exist
            severity = self.extreme_handler.analyze_lighting_severity(image)
            
            if severity['is_extreme']:
                t0 = time.time()
                
                # Apply extreme lighting handler
                image = self.extreme_handler.process(image)
                
                # Apply local adaptive if very extreme
                if severity['dynamic_range'] > 200 and self.local_adaptive:
                    image = self.local_adaptive.process(image)
                
                # Apply shadow/highlight recovery
                if self.shadow_highlight and (severity['has_specular'] or severity['has_deep_shadows']):
                    image = self.shadow_highlight.process(image)
                
                timings['advanced_illumination'] = (time.time() - t0) * 1000
                intermediates['advanced_illumination'] = image.copy()
                logger.info(f"Advanced illumination in {timings['advanced_illumination']:.1f}ms")
            
            # Handle edge cases
            processed, edge_info = self.edge_handler.handle_edge_cases(image)
            if edge_info['severity'] != 'none':
                image = processed
                logger.info(f"Handled edge cases: {edge_info['severity']}")
        
        # Step 4: Segmentation (300-500ms)
        if self.enable_segmentation and self.segmentation is not None:
            t0 = time.time()
            # Segment the image
            mask, regions = self.segmentation.segment(image)
            # Apply mask to the image
            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) / 255.0
            image = (image * mask_3ch).astype(np.uint8)
            timings['segmentation'] = (time.time() - t0) * 1000
            intermediates['segmented'] = image.copy()
            logger.info(f"Segmentation completed in {timings['segmentation']:.1f}ms")
        
        # Step 5: Color Space Fusion (50-100ms)
        t0 = time.time()
        color_features = self._extract_color_features(image)
        timings['color_fusion'] = (time.time() - t0) * 1000
        intermediates['color_features'] = color_features
        logger.info(f"Color fusion in {timings['color_fusion']:.1f}ms")
        
        # Step 6: Vegetation Indices (20-50ms)
        t0 = time.time()
        indices = self._calculate_vegetation_indices(image)
        timings['indices'] = (time.time() - t0) * 1000
        intermediates['indices'] = indices
        logger.info(f"Vegetation indices in {timings['indices']:.1f}ms")
        
        # Total time
        total_time = (time.time() - start_time) * 1000
        logger.info(f"Total preprocessing time: {total_time:.1f}ms")
        
        # Update statistics
        self._update_stats(timings)
        
        if return_intermediate:
            return {
                'final': image,
                'intermediates': intermediates,
                'timings': timings,
                'total_time_ms': total_time,
                'color_features': color_features,
                'vegetation_indices': indices
            }
        
        return image
    
    def _apply_multi_resolution(self, 
                               image: np.ndarray, 
                               target_size: Tuple[int, int]) -> np.ndarray:
        """
        Apply multi-resolution processing
        Optimal sizes: 384x384 for balance, 512x512 for max accuracy
        """
        h, w = image.shape[:2]
        target_h, target_w = target_size
        
        # Determine if we need to resize
        if h != target_h or w != target_w:
            # Use different interpolation based on up/downscaling
            if h < target_h or w < target_w:
                # Upscaling - use cubic
                interpolation = cv2.INTER_CUBIC
            else:
                # Downscaling - use area
                interpolation = cv2.INTER_AREA
            
            image = cv2.resize(image, (target_w, target_h), interpolation=interpolation)
        
        return image
    
    def _apply_illumination_normalization(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Retinex-based illumination normalization
        Handles lighting variance from direct sun to shade
        Reduces variance from 35% to <10% while preserving disease patterns
        """
        if self.illumination is not None:
            # Use full Retinex implementation
            return self.illumination.process(
                image,
                preserve_disease=True,
                use_multi_scale=True
            )
        else:
            # Fallback to simple normalization
            return correct_exposure(image)
    
    def _extract_color_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract features from multiple color spaces
        RGB, LAB, HSV, YCbCr for comprehensive disease detection
        """
        features = {}
        
        # RGB features (already have)
        features['rgb'] = image
        
        # LAB for perceptual differences
        features['lab'] = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # HSV for hue-based disease detection
        features['hsv'] = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # YCbCr for luminance separation
        features['ycrcb'] = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        
        # Extract statistical features from each color space
        stats = {}
        for space_name, space_img in features.items():
            stats[f'{space_name}_mean'] = np.mean(space_img, axis=(0, 1))
            stats[f'{space_name}_std'] = np.std(space_img, axis=(0, 1))
        
        features['statistics'] = stats
        
        return features
    
    def _calculate_vegetation_indices(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate vegetation indices for disease detection
        VARI, MGRVI, vNDVI
        """
        # Ensure float for calculations
        img_float = image.astype(np.float32) / 255.0
        
        # Extract channels
        r = img_float[:, :, 0]
        g = img_float[:, :, 1]
        b = img_float[:, :, 2]
        
        # Avoid division by zero
        eps = 1e-10
        
        indices = {}
        
        # VARI (Visible Atmospherically Resistant Index)
        # (Green - Red) / (Green + Red - Blue)
        denominator = g + r - b + eps
        vari = np.where(denominator != 0, (g - r) / denominator, 0)
        indices['vari'] = vari
        
        # MGRVI (Modified Green Red Vegetation Index)
        # (Green² - Red²) / (Green² + Red²)
        g_squared = g ** 2
        r_squared = r ** 2
        denominator = g_squared + r_squared + eps
        mgrvi = (g_squared - r_squared) / denominator
        indices['mgrvi'] = mgrvi
        
        # vNDVI (visible Normalized Difference Vegetation Index)
        # Approximation using visible bands
        # (NIR - Red) / (NIR + Red), but we approximate NIR with Green
        denominator = g + r + eps
        vndvi = (g - r) / denominator
        indices['vndvi'] = vndvi
        
        # Combine indices into a feature map
        combined = np.stack([vari, mgrvi, vndvi], axis=-1)
        indices['combined'] = combined
        
        return indices
    
    def _update_stats(self, timings: Dict[str, float]):
        """Update pipeline statistics"""
        self.stats['total_processed'] += 1
        n = self.stats['total_processed']
        
        # Update average total time
        total_time = sum(timings.values())
        prev_avg = self.stats['avg_time_ms']
        self.stats['avg_time_ms'] = (prev_avg * (n - 1) + total_time) / n
        
        # Update component times
        for component, time_ms in timings.items():
            if component in self.stats['component_times']:
                prev = self.stats['component_times'][component]
                self.stats['component_times'][component] = (prev * (n - 1) + time_ms) / n
    
    def get_stats(self) -> Dict:
        """Get pipeline statistics"""
        return self.stats.copy()
    
    def validate_performance(self, test_images: List[np.ndarray]) -> Dict:
        """
        Validate pipeline performance against targets
        
        Expected targets:
        - Total time: 400-600ms
        - LASSR: 200-400ms
        - Segmentation: 300-500ms
        - Illumination: 150-200ms
        """
        results = {
            'times': [],
            'component_times': {comp: [] for comp in self.stats['component_times']},
            'meets_targets': True
        }
        
        for image in test_images:
            result = self.process(image, return_intermediate=True)
            results['times'].append(result['total_time_ms'])
            
            for component, time_ms in result['timings'].items():
                if component in results['component_times']:
                    results['component_times'][component].append(time_ms)
        
        # Calculate averages
        results['avg_total_ms'] = np.mean(results['times'])
        results['avg_components'] = {
            comp: np.mean(times) if times else 0
            for comp, times in results['component_times'].items()
        }
        
        # Check targets
        if results['avg_total_ms'] > 600:
            results['meets_targets'] = False
            results['issue'] = f"Total time {results['avg_total_ms']:.0f}ms exceeds 600ms"
        
        if 'lassr' in results['avg_components'] and results['avg_components']['lassr'] > 400:
            results['meets_targets'] = False
            results['issue'] = f"LASSR time {results['avg_components']['lassr']:.0f}ms exceeds 400ms"
        
        return results


def create_default_pipeline() -> PreprocessingPipeline:
    """
    Create default preprocessing pipeline with optimal settings
    """
    return PreprocessingPipeline(
        enable_lassr=True,
        enable_segmentation=True,  # Will be enabled when implemented
        enable_illumination=True,
        device='auto'
    )


def process_iphone_image(image_path: str,
                        pipeline: Optional[PreprocessingPipeline] = None) -> Tuple[np.ndarray, Dict]:
    """
    Process iPhone HEIC image with metadata-aware processing
    
    Args:
        image_path: Path to HEIC/HEIF image
        pipeline: Optional pipeline instance
        
    Returns:
        Tuple of (processed_image, metadata)
    """
    if pipeline is None:
        pipeline = create_default_pipeline()
    
    # Load HEIC with metadata
    if pipeline.heic_processor:
        image, metadata = pipeline.heic_processor.process(image_path)
        
        # Process with pipeline
        processed = pipeline.process(image, target_size=(384, 384))
        
        return processed, metadata
    else:
        # Fallback to regular processing
        processed = pipeline.process(image_path, target_size=(384, 384))
        return processed, {}


def process_for_disease_detection(image_path: str,
                                 save_preprocessed: bool = False,
                                 output_path: Optional[str] = None) -> np.ndarray:
    """
    Convenience function to process an image for disease detection
    
    Args:
        image_path: Path to input image
        save_preprocessed: Whether to save the preprocessed image
        output_path: Where to save if save_preprocessed is True
    
    Returns:
        Preprocessed image ready for disease detection
    """
    pipeline = create_default_pipeline()
    
    # Process image
    result = pipeline.process(image_path, target_size=(384, 384))
    
    # Save if requested
    if save_preprocessed:
        if output_path is None:
            output_path = str(Path(image_path).with_suffix('.preprocessed.jpg'))
        cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        logger.info(f"Saved preprocessed image to {output_path}")
    
    return result