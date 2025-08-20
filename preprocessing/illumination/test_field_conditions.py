"""
Field Condition Testing Suite
Tests the complete illumination system under simulated field conditions
Validates that we're closing the 79% sun vs 89% indoor accuracy gap
"""

import sys
import os
from pathlib import Path
import numpy as np
import cv2
import time
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from preprocessing.illumination.retinex_illumination import RetinexIllumination
from preprocessing.illumination.extreme_conditions import ExtremeLightingHandler
from preprocessing.illumination.local_adaptive import LocalAdaptiveProcessor
from preprocessing.illumination.shadow_highlight import ShadowHighlightRecovery
from preprocessing.illumination.edge_cases import EdgeCaseHandler
from preprocessing.illumination.disease_pattern_generator import DiseasePatternGenerator
from preprocessing.heic_handler import HEICProcessor


class FieldConditionTester:
    """
    Comprehensive testing for field conditions
    Simulates iPhone camera characteristics and real-world scenarios
    """
    
    def __init__(self):
        """Initialize field condition tester"""
        # Initialize all processors
        self.retinex = RetinexIllumination()
        self.extreme_handler = ExtremeLightingHandler()
        self.local_adaptive = LocalAdaptiveProcessor()
        self.shadow_highlight = ShadowHighlightRecovery()
        self.edge_handler = EdgeCaseHandler()
        self.heic_processor = HEICProcessor()
        self.generator = DiseasePatternGenerator()
        
        # Test results storage
        self.results = {
            'conditions': {},
            'performance': {},
            'accuracy_gap': {},
            'edge_cases': {}
        }
    
    def run_complete_test_suite(self):
        """Run all field condition tests"""
        print("\n" + "="*70)
        print("FIELD CONDITION TEST SUITE - DAY 3 ENHANCEMENTS")
        print("Target: Reduce sun/indoor accuracy gap from 10% to <5%")
        print("="*70)
        
        # Test 1: Extreme lighting conditions
        print("\n1. Testing Extreme Lighting Conditions...")
        self.test_extreme_lighting()
        
        # Test 2: Local adaptive processing
        print("\n2. Testing Local Adaptive Processing...")
        self.test_local_adaptive()
        
        # Test 3: iPhone-specific scenarios
        print("\n3. Testing iPhone-Specific Scenarios...")
        self.test_iphone_scenarios()
        
        # Test 4: Edge cases
        print("\n4. Testing Edge Cases...")
        self.test_edge_cases()
        
        # Test 5: Performance benchmarks
        print("\n5. Testing Performance...")
        self.test_performance()
        
        # Test 6: Accuracy gap analysis
        print("\n6. Analyzing Accuracy Gap...")
        self.analyze_accuracy_gap()
        
        # Print summary
        self.print_test_summary()
    
    def test_extreme_lighting(self):
        """Test extreme lighting handler"""
        conditions = [
            ('harsh_sun', 'overexposed'),  # 79% accuracy scenario
            ('deep_shade', 'underexposed'),
            ('mixed_sun_shade', 'harsh_shadow'),
            ('golden_hour', 'backlit')
        ]
        
        for name, condition in conditions:
            print(f"\n   Testing {name}...")
            
            # Create disease pattern
            base = self.generator.create_healthy_leaf((512, 512))
            disease = self.generator.create_blight_pattern(base, severity='moderate')
            
            # Apply lighting condition
            distorted = self.generator.apply_lighting_condition(disease, condition)
            
            # Process with extreme lighting handler
            start_time = time.time()
            recovered = self.extreme_handler.process(distorted)
            elapsed_ms = (time.time() - start_time) * 1000
            
            # Measure improvement
            disease_score_before = self.measure_disease_visibility(distorted, disease.mask)
            disease_score_after = self.measure_disease_visibility(recovered, disease.mask)
            
            improvement = disease_score_after - disease_score_before
            
            print(f"     Disease visibility: {disease_score_before:.2f} -> {disease_score_after:.2f}")
            print(f"     Improvement: {improvement:+.2f}")
            print(f"     Processing time: {elapsed_ms:.1f}ms")
            
            self.results['conditions'][name] = {
                'before': disease_score_before,
                'after': disease_score_after,
                'improvement': improvement,
                'time_ms': elapsed_ms
            }
    
    def test_local_adaptive(self):
        """Test local adaptive processing"""
        # Create image with varying lighting regions
        image = self.create_mixed_lighting_image()
        
        print("   Processing image with 4x4 grid adaptation...")
        
        start_time = time.time()
        processed = self.local_adaptive.process(image)
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Analyze uniformity improvement
        uniformity_before = self.measure_lighting_uniformity(image)
        uniformity_after = self.measure_lighting_uniformity(processed)
        
        print(f"   Lighting uniformity: {uniformity_before:.2f} -> {uniformity_after:.2f}")
        print(f"   Processing time: {elapsed_ms:.1f}ms")
        
        self.results['performance']['local_adaptive'] = {
            'uniformity_improvement': uniformity_after - uniformity_before,
            'time_ms': elapsed_ms
        }
    
    def test_iphone_scenarios(self):
        """Test iPhone-specific scenarios"""
        scenarios = [
            {
                'name': 'iPhone_harsh_sun',
                'metadata': {
                    'iso': 32,
                    'exposure_time': (1, 2000),
                    'model': 'iPhone 14 Pro',
                    'has_metadata': True
                },
                'condition': 'overexposed'
            },
            {
                'name': 'iPhone_low_light',
                'metadata': {
                    'iso': 1600,
                    'exposure_time': (1, 30),
                    'model': 'iPhone 14 Pro',
                    'has_metadata': True
                },
                'condition': 'underexposed'
            },
            {
                'name': 'iPhone_flash',
                'metadata': {
                    'iso': 100,
                    'flash': 1,
                    'model': 'iPhone 14 Pro',
                    'has_metadata': True
                },
                'condition': 'overexposed'
            }
        ]
        
        for scenario in scenarios:
            print(f"\n   Testing {scenario['name']}...")
            
            # Create test image
            base = self.generator.create_healthy_leaf((512, 512))
            disease = self.generator.create_powdery_mildew_pattern(base, severity='moderate')
            distorted = self.generator.apply_lighting_condition(disease, scenario['condition'])
            
            # Get adaptive parameters from metadata
            params = self.heic_processor.get_adaptive_parameters(scenario['metadata'])
            
            # Process with metadata-aware parameters
            processed = self.heic_processor.preprocess_iphone_image(distorted, scenario['metadata'])
            
            # Apply extreme lighting handler
            final = self.extreme_handler.process(processed)
            
            # Measure results
            disease_visibility = self.measure_disease_visibility(final, disease.mask)
            
            print(f"     Adaptive params: gamma={params['gamma']:.1f}, clahe={params['clahe_clip']:.1f}")
            print(f"     Disease visibility: {disease_visibility:.2f}")
            
            self.results['conditions'][scenario['name']] = {
                'disease_visibility': disease_visibility,
                'params': params
            }
    
    def test_edge_cases(self):
        """Test edge case handling"""
        edge_cases = [
            ('blown_out', self.create_blown_out_image),
            ('extremely_dark', self.create_extremely_dark_image),
            ('motion_blur', self.create_motion_blur_image),
            ('mixed_lighting', self.create_mixed_lighting_image)
        ]
        
        for name, create_func in edge_cases:
            print(f"\n   Testing {name}...")
            
            # Create edge case image
            image = create_func()
            
            # Process with edge case handler
            start_time = time.time()
            processed, edge_info = self.edge_handler.handle_edge_cases(image)
            elapsed_ms = (time.time() - start_time) * 1000
            
            # Validate output
            is_valid = self.edge_handler.validate_output(processed)
            
            print(f"     Detected issues: {edge_info['severity']}")
            print(f"     Output valid: {is_valid}")
            print(f"     Processing time: {elapsed_ms:.1f}ms")
            
            self.results['edge_cases'][name] = {
                'severity': edge_info['severity'],
                'valid': is_valid,
                'time_ms': elapsed_ms
            }
    
    def test_performance(self):
        """Test overall performance with all enhancements"""
        print("\n   Testing complete pipeline performance...")
        
        sizes = [(256, 256), (384, 384), (512, 512)]
        
        for size in sizes:
            # Create test image
            base = self.generator.create_healthy_leaf(size)
            disease = self.generator.create_mosaic_virus_pattern(base)
            harsh_sun = self.generator.apply_lighting_condition(disease, 'overexposed')
            
            # Complete processing pipeline
            times = {}
            
            # Retinex
            start = time.time()
            step1 = self.retinex.process(harsh_sun)
            times['retinex'] = (time.time() - start) * 1000
            
            # Extreme lighting
            start = time.time()
            step2 = self.extreme_handler.process(step1)
            times['extreme'] = (time.time() - start) * 1000
            
            # Local adaptive
            start = time.time()
            step3 = self.local_adaptive.process(step2)
            times['local_adaptive'] = (time.time() - start) * 1000
            
            # Shadow/highlight
            start = time.time()
            final = self.shadow_highlight.process(step3)
            times['shadow_highlight'] = (time.time() - start) * 1000
            
            total_time = sum(times.values())
            
            print(f"\n   Size {size[0]}x{size[1]}:")
            print(f"     Retinex: {times['retinex']:.1f}ms")
            print(f"     Extreme: {times['extreme']:.1f}ms")
            print(f"     Local Adaptive: {times['local_adaptive']:.1f}ms")
            print(f"     Shadow/Highlight: {times['shadow_highlight']:.1f}ms")
            print(f"     Total: {total_time:.1f}ms")
            
            if size == (384, 384):
                self.results['performance']['total_time_ms'] = total_time
                self.results['performance']['breakdown'] = times
    
    def analyze_accuracy_gap(self):
        """Analyze sun vs indoor accuracy gap"""
        # Simulate accuracy under different conditions
        conditions = {
            'direct_sun': [],
            'indoor': [],
            'cloudy': [],
            'shade': []
        }
        
        # Test multiple disease patterns under each condition
        diseases = ['blight', 'leaf_spot', 'powdery_mildew', 'mosaic_virus']
        
        for disease_type in diseases:
            for condition_name in conditions.keys():
                # Create disease
                base = self.generator.create_healthy_leaf((384, 384))
                
                if disease_type == 'blight':
                    disease = self.generator.create_blight_pattern(base)
                elif disease_type == 'leaf_spot':
                    disease = self.generator.create_leaf_spot_pattern(base)
                elif disease_type == 'powdery_mildew':
                    disease = self.generator.create_powdery_mildew_pattern(base)
                else:
                    disease = self.generator.create_mosaic_virus_pattern(base)
                
                # Apply condition
                if condition_name == 'direct_sun':
                    distorted = self.generator.apply_lighting_condition(disease, 'overexposed')
                elif condition_name == 'indoor':
                    distorted = disease.image  # Indoor is "normal"
                elif condition_name == 'cloudy':
                    distorted = self.generator.apply_lighting_condition(disease, 'underexposed')
                else:  # shade
                    distorted = self.generator.apply_lighting_condition(disease, 'harsh_shadow')
                
                # Process with complete pipeline
                processed = self.process_complete_pipeline(distorted)
                
                # Measure disease preservation
                score = self.measure_disease_visibility(processed, disease.mask)
                conditions[condition_name].append(score)
        
        # Calculate average scores
        avg_scores = {k: np.mean(v) for k, v in conditions.items()}
        
        # Calculate gap
        sun_score = avg_scores['direct_sun']
        indoor_score = avg_scores['indoor']
        gap = indoor_score - sun_score
        
        print(f"\n   Direct Sun: {sun_score:.3f}")
        print(f"   Indoor: {indoor_score:.3f}")
        print(f"   Cloudy: {avg_scores['cloudy']:.3f}")
        print(f"   Shade: {avg_scores['shade']:.3f}")
        print(f"\n   Sun/Indoor Gap: {gap:.3f} ({gap*100:.1f}%)")
        
        self.results['accuracy_gap'] = {
            'sun': sun_score,
            'indoor': indoor_score,
            'gap': gap,
            'gap_percentage': gap * 100
        }
    
    def process_complete_pipeline(self, image: np.ndarray) -> np.ndarray:
        """Process image through complete Day 3 pipeline"""
        # Retinex base processing
        result = self.retinex.process(image)
        
        # Extreme lighting handling
        result = self.extreme_handler.process(result)
        
        # Local adaptive (only if needed)
        severity = self.extreme_handler.analyze_lighting_severity(result)
        if severity['is_extreme']:
            result = self.local_adaptive.process(result)
        
        # Shadow/highlight recovery
        result = self.shadow_highlight.process(result)
        
        # Edge case handling
        result, _ = self.edge_handler.handle_edge_cases(result)
        
        return result
    
    def measure_disease_visibility(self, image: np.ndarray, mask: np.ndarray) -> float:
        """Measure how visible disease patterns are"""
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Calculate contrast between disease and background
        disease_pixels = gray[mask > 0]
        background_pixels = gray[mask == 0]
        
        if len(disease_pixels) == 0 or len(background_pixels) == 0:
            return 0.0
        
        # Multiple metrics for visibility
        contrast = abs(np.mean(disease_pixels) - np.mean(background_pixels))
        edge_strength = np.std(disease_pixels)
        
        # Normalize and combine
        visibility_score = min((contrast / 50.0) * 0.7 + (edge_strength / 30.0) * 0.3, 1.0)
        
        return visibility_score
    
    def measure_lighting_uniformity(self, image: np.ndarray) -> float:
        """Measure lighting uniformity across image"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Divide into grid
        h, w = gray.shape
        grid_h, grid_w = h // 4, w // 4
        
        region_means = []
        for i in range(4):
            for j in range(4):
                region = gray[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]
                region_means.append(np.mean(region))
        
        # Lower variance = more uniform
        uniformity = 1.0 / (1.0 + np.std(region_means) / 100.0)
        
        return uniformity
    
    def create_blown_out_image(self) -> np.ndarray:
        """Create severely overexposed test image"""
        base = self.generator.create_healthy_leaf((384, 384))
        disease = self.generator.create_leaf_spot_pattern(base)
        
        # Extreme overexposure
        blown = disease.image.astype(np.float32) * 3.0 + 150
        return np.clip(blown, 0, 255).astype(np.uint8)
    
    def create_extremely_dark_image(self) -> np.ndarray:
        """Create extremely dark test image"""
        base = self.generator.create_healthy_leaf((384, 384))
        disease = self.generator.create_blight_pattern(base)
        
        # Extreme underexposure
        dark = disease.image.astype(np.float32) * 0.1
        return np.clip(dark, 0, 255).astype(np.uint8)
    
    def create_motion_blur_image(self) -> np.ndarray:
        """Create motion blurred test image"""
        base = self.generator.create_healthy_leaf((384, 384))
        disease = self.generator.create_mosaic_virus_pattern(base)
        
        # Apply motion blur
        kernel_size = 15
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size//2, :] = 1.0 / kernel_size
        
        blurred = cv2.filter2D(disease.image, -1, kernel)
        return blurred
    
    def create_mixed_lighting_image(self) -> np.ndarray:
        """Create image with mixed lighting"""
        base = self.generator.create_healthy_leaf((384, 384))
        disease = self.generator.create_powdery_mildew_pattern(base)
        
        # Half bright, half dark
        mixed = disease.image.copy()
        h, w = mixed.shape[:2]
        
        # Left half dark
        mixed[:, :w//2] = (mixed[:, :w//2] * 0.3).astype(np.uint8)
        
        # Right half bright
        mixed[:, w//2:] = np.clip(mixed[:, w//2:] * 1.8 + 50, 0, 255).astype(np.uint8)
        
        return mixed
    
    def print_test_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "="*70)
        print("FIELD CONDITION TEST SUMMARY")
        print("="*70)
        
        # Performance summary
        if 'total_time_ms' in self.results['performance']:
            total_time = self.results['performance']['total_time_ms']
            print(f"\nTotal Processing Time: {total_time:.1f}ms")
            
            if total_time < 400:
                print("  [OK] Well within 400-600ms target")
            elif total_time < 600:
                print("  [OK] Within 400-600ms target")
            else:
                print("  [WARN] Exceeds 600ms target")
        
        # Accuracy gap summary
        if 'accuracy_gap' in self.results:
            gap = self.results['accuracy_gap']['gap_percentage']
            print(f"\nSun/Indoor Accuracy Gap: {gap:.1f}%")
            
            if gap < 5:
                print("  [SUCCESS] Gap reduced to <5% (target achieved!)")
            elif gap < 7:
                print("  [OK] Gap reduced significantly")
            else:
                print("  [NEEDS WORK] Gap still >7%")
        
        # Edge case handling
        edge_success = sum(1 for v in self.results['edge_cases'].values() if v['valid'])
        edge_total = len(self.results['edge_cases'])
        
        print(f"\nEdge Cases Handled: {edge_success}/{edge_total}")
        
        # Disease visibility scores
        avg_visibility = np.mean([v.get('after', 0) for v in self.results['conditions'].values()])
        print(f"\nAverage Disease Visibility: {avg_visibility:.2f}")
        
        if avg_visibility > 0.9:
            print("  [EXCELLENT] Disease patterns highly preserved")
        elif avg_visibility > 0.8:
            print("  [GOOD] Disease patterns well preserved")
        else:
            print("  [NEEDS IMPROVEMENT] Disease preservation could be better")
        
        print("\n" + "="*70)
        print("Day 3 Advanced Illumination Complete!")
        print("Ready for Day 4 Segmentation Integration")
        print("="*70)


if __name__ == "__main__":
    tester = FieldConditionTester()
    tester.run_complete_test_suite()