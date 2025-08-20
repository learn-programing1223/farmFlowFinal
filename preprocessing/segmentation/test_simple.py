"""
Simple test to verify basic segmentation functionality
"""

import sys
from pathlib import Path
import numpy as np
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from preprocessing.segmentation.disease_detector import DiseaseRegionDetector
from preprocessing.segmentation.rgb_segmentation import RGBSegmentation
from preprocessing.segmentation.disease_protection import DiseaseProtectionLayer
from preprocessing.segmentation.cascade_controller import SegmentationCascade
from preprocessing.illumination.disease_pattern_generator import DiseasePatternGenerator


def test_basic_segmentation():
    """Test basic segmentation pipeline"""
    print("\n" + "="*60)
    print("BASIC SEGMENTATION TEST")
    print("="*60)
    
    # Initialize components
    generator = DiseasePatternGenerator()
    cascade = SegmentationCascade(enable_deeplab=False)
    
    # Create test image with disease
    print("\n1. Creating test image with blight...")
    base = generator.create_healthy_leaf((384, 384))
    disease = generator.create_blight_pattern(base, severity='moderate')
    
    print(f"   Disease coverage: {np.sum(disease.mask > 0) / disease.mask.size:.2%}")
    
    # Test disease detection
    print("\n2. Testing disease detection...")
    detector = DiseaseRegionDetector(sensitivity=0.8)
    
    start = time.time()
    disease_mask, info = detector.detect(disease.image)
    elapsed = (time.time() - start) * 1000
    
    print(f"   Detection time: {elapsed:.1f}ms")
    print(f"   Detected coverage: {info['total_coverage']:.2%}")
    print(f"   Brown regions: {info['brown_regions']}")
    print(f"   Dark spots: {info['dark_spots']}")
    
    # Test RGB segmentation
    print("\n3. Testing RGB segmentation...")
    rgb_segmenter = RGBSegmentation()
    
    start = time.time()
    plant_mask, seg_info = rgb_segmenter.segment(disease.image, disease_mask)
    elapsed = (time.time() - start) * 1000
    
    print(f"   Segmentation time: {elapsed:.1f}ms")
    print(f"   Coverage: {seg_info['coverage']:.2%}")
    print(f"   Indices used: {seg_info['indices_used']}")
    
    # Test disease protection
    print("\n4. Testing disease protection...")
    protector = DiseaseProtectionLayer()
    
    # Create a partial plant mask (simulate incomplete segmentation)
    partial_mask = plant_mask.copy()
    partial_mask[200:300, 200:300] = 0  # Remove part
    
    protected_mask, prot_info = protector.protect(partial_mask, disease_mask, validate=True)
    
    print(f"   Initial coverage: {prot_info['initial_coverage']:.2%}")
    print(f"   Final coverage: {prot_info['final_coverage']:.2%}")
    print(f"   Disease preserved: {prot_info['disease_preserved']:.2%}")
    print(f"   Validation passed: {prot_info['validation_passed']}")
    
    # Test full cascade
    print("\n5. Testing complete cascade...")
    
    # Use simpler approach - just the disease image directly
    start = time.time()
    final_mask, cascade_info = cascade.segment(disease.image, use_fallback=False)
    elapsed = (time.time() - start) * 1000
    
    print(f"   Cascade time: {elapsed:.1f}ms")
    print(f"   Path used: {cascade_info['path_used']}")
    print(f"   Disease detected: {cascade_info['disease_detected']}")
    print(f"   Quality score: {cascade_info['quality_score']:.2f}")
    
    # Validate disease preservation
    if np.sum(disease.mask) > 0:
        preserved = np.sum(np.logical_and(final_mask > 0, disease.mask > 0))
        total_disease = np.sum(disease.mask > 0)
        preservation = preserved / total_disease
        print(f"   Final disease preservation: {preservation:.2%}")
    
    print("\n" + "="*60)
    print("Basic test complete!")
    print("="*60)


if __name__ == "__main__":
    test_basic_segmentation()