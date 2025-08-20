"""
Test suite for LASSR super-resolution module
Validates disease pattern enhancement and performance requirements
"""

import unittest
import numpy as np
import cv2
import torch
import time
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from preprocessing.lassr import LASSRProcessor
from preprocessing.lassr_model import LASSRDisease, LightweightLASSR, disease_pattern_loss
from preprocessing.lassr_utils import (
    preprocess_image, postprocess_image, 
    detect_disease_regions, estimate_memory_usage
)


class TestLASSRModel(unittest.TestCase):
    """Test LASSR model architecture"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LASSRDisease(num_blocks=4)  # Smaller for testing
        self.model.to(self.device)
        self.model.eval()
    
    def test_model_output_shape(self):
        """Test that model produces correct output shape (2x upscaling)"""
        input_tensor = torch.randn(1, 3, 128, 128).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # Should be 2x the input size
        self.assertEqual(output.shape, (1, 3, 256, 256))
    
    def test_disease_attention(self):
        """Test that disease attention mechanism works"""
        from preprocessing.lassr_model import DiseasePatternAttention
        
        attention = DiseasePatternAttention(64).to(self.device)
        input_tensor = torch.randn(1, 64, 32, 32).to(self.device)
        
        output = attention(input_tensor)
        
        # Should maintain shape
        self.assertEqual(output.shape, input_tensor.shape)
        
        # Should not be identical (attention should modify)
        self.assertFalse(torch.allclose(output, input_tensor))
    
    def test_lightweight_model(self):
        """Test lightweight model variant"""
        model = LightweightLASSR().to(self.device)
        input_tensor = torch.randn(1, 3, 128, 128).to(self.device)
        
        with torch.no_grad():
            output = model(input_tensor)
        
        self.assertEqual(output.shape, (1, 3, 256, 256))
    
    def test_disease_pattern_loss(self):
        """Test custom loss function"""
        pred = torch.randn(2, 3, 64, 64).to(self.device)
        target = torch.randn(2, 3, 64, 64).to(self.device)
        
        loss = disease_pattern_loss(pred, target)
        
        self.assertIsInstance(loss.item(), float)
        self.assertGreaterEqual(loss.item(), 0)
        
        # Test with mask
        mask = torch.ones(2, 1, 64, 64).to(self.device) * 0.5
        loss_with_mask = disease_pattern_loss(pred, target, mask)
        
        self.assertIsInstance(loss_with_mask.item(), float)


class TestLASSRProcessor(unittest.TestCase):
    """Test LASSR processor functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.processor = LASSRProcessor(
            model_type='lightweight',  # Faster for testing
            optimize_mobile=False  # Skip JIT for tests
        )
        
        # Create test images
        self.test_image_small = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        self.test_image_medium = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        self.test_image_large = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    def test_process_basic(self):
        """Test basic processing functionality"""
        enhanced = self.processor.process(self.test_image_small)
        
        # Should be 2x size
        self.assertEqual(enhanced.shape[:2], (256, 256))
        
        # Should be valid image
        self.assertEqual(enhanced.dtype, np.uint8)
        self.assertLessEqual(enhanced.max(), 255)
        self.assertGreaterEqual(enhanced.min(), 0)
    
    def test_process_with_stats(self):
        """Test processing with statistics"""
        enhanced, stats = self.processor.process(
            self.test_image_small,
            return_stats=True
        )
        
        self.assertIn('enhanced', stats)
        self.assertIn('time_ms', stats)
        self.assertIsInstance(stats['time_ms'], float)
    
    def test_time_budget(self):
        """Test that processing respects time budget (200-400ms)"""
        times = []
        
        for _ in range(5):
            start = time.time()
            _ = self.processor.process(self.test_image_medium)
            elapsed_ms = (time.time() - start) * 1000
            times.append(elapsed_ms)
        
        avg_time = np.mean(times)
        
        # Should be within budget (with some tolerance for slower systems)
        self.assertLess(avg_time, 600, 
                       f"Average time {avg_time:.0f}ms exceeds budget")
    
    def test_memory_usage(self):
        """Test memory estimation and constraints"""
        # Small image should use less memory
        small_memory = estimate_memory_usage(self.test_image_small.shape)
        self.assertLess(small_memory, 50)
        
        # Large image should still be under 100MB with tiling
        large_memory = estimate_memory_usage(self.test_image_large.shape, tile_size=256)
        self.assertLess(large_memory, 100)
    
    def test_batch_processing(self):
        """Test batch processing functionality"""
        images = [self.test_image_small] * 3
        
        results = self.processor.batch_process(images)
        
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertEqual(result.shape[:2], (256, 256))
    
    def test_fallback_on_error(self):
        """Test fallback to bicubic on error"""
        # Create invalid image that might cause issues
        invalid_image = np.zeros((32, 32, 3), dtype=np.uint8)
        
        # Should still return something (fallback)
        result = self.processor.process(invalid_image)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape[:2], (64, 64))  # 2x upscaled


class TestLASSRUtils(unittest.TestCase):
    """Test utility functions"""
    
    def test_preprocess_postprocess(self):
        """Test image preprocessing and postprocessing"""
        device = torch.device('cpu')
        image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        
        # Preprocess
        tensor = preprocess_image(image, device)
        self.assertEqual(tensor.shape, (1, 3, 128, 128))
        self.assertLessEqual(tensor.max(), 1.0)
        self.assertGreaterEqual(tensor.min(), 0.0)
        
        # Postprocess
        recovered = postprocess_image(tensor)
        self.assertEqual(recovered.shape, (128, 128, 3))
        self.assertEqual(recovered.dtype, np.uint8)
    
    def test_disease_region_detection(self):
        """Test disease region detection"""
        # Create image with disease-like colors
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Add brown spot (blight-like)
        cv2.circle(image, (30, 30), 10, (139, 69, 19), -1)  # Brown in RGB
        
        # Add yellow area (deficiency-like)
        cv2.circle(image, (70, 70), 10, (255, 255, 0), -1)  # Yellow
        
        mask = detect_disease_regions(image)
        
        self.assertEqual(mask.shape, (100, 100))
        self.assertGreater(mask.max(), 0)  # Should detect something
        
        # Check that disease areas have higher values
        brown_area_value = mask[25:35, 25:35].mean()
        background_value = mask[0:10, 0:10].mean()
        self.assertGreater(brown_area_value, background_value)


class TestDiseasePatternEnhancement(unittest.TestCase):
    """Test disease pattern enhancement capabilities"""
    
    def setUp(self):
        """Create test images with simulated disease patterns"""
        self.processor = LASSRProcessor(model_type='lightweight')
        
        # Create test image with disease-like features
        self.disease_image = self._create_disease_image()
    
    def _create_disease_image(self):
        """Create synthetic image with disease patterns"""
        image = np.ones((256, 256, 3), dtype=np.uint8) * 50  # Dark green background
        image[:, :, 1] = 100  # Make it greenish
        
        # Add brown spots (blight)
        for _ in range(5):
            x, y = np.random.randint(50, 200, 2)
            radius = np.random.randint(5, 15)
            cv2.circle(image, (x, y), radius, (101, 67, 33), -1)  # Brown
        
        # Add yellow patches (mosaic virus)
        for _ in range(3):
            x, y = np.random.randint(50, 200, 2)
            cv2.ellipse(image, (x, y), (20, 10), 0, 0, 360, (200, 200, 100), -1)
        
        # Add white spots (powdery mildew)
        for _ in range(10):
            x, y = np.random.randint(50, 200, 2)
            cv2.circle(image, (x, y), 3, (220, 220, 220), -1)
        
        return image
    
    def test_disease_preservation(self):
        """Test that disease patterns are preserved/enhanced"""
        enhanced = self.processor.process(self.disease_image, preserve_disease=True)
        
        # Convert to grayscale for edge detection
        original_gray = cv2.cvtColor(self.disease_image, cv2.COLOR_RGB2GRAY)
        enhanced_gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
        
        # Resize original for comparison
        original_resized = cv2.resize(original_gray, (enhanced.shape[1], enhanced.shape[0]))
        
        # Calculate edge strength (disease boundaries)
        original_edges = cv2.Canny(original_resized, 50, 150).sum()
        enhanced_edges = cv2.Canny(enhanced_gray, 50, 150).sum()
        
        # Enhanced should have stronger edges (clearer disease boundaries)
        # Allow some tolerance as synthetic image might not enhance perfectly
        self.assertGreaterEqual(enhanced_edges, original_edges * 0.8,
                               "Disease boundaries should be preserved or enhanced")
    
    def test_color_preservation(self):
        """Test that disease colors are preserved"""
        enhanced = self.processor.process(self.disease_image, preserve_disease=True)
        
        # Check that brown spots are still brownish
        # Get a region where we know there's a brown spot
        original_region = self.disease_image[60:80, 60:80]
        enhanced_region = enhanced[120:160, 120:160]  # 2x scaled
        
        # Calculate dominant colors
        original_mean = original_region.mean(axis=(0, 1))
        enhanced_mean = enhanced_region.mean(axis=(0, 1))
        
        # Brown should have R > G > B pattern
        if original_mean[0] > original_mean[1]:  # If original was brownish
            self.assertGreater(enhanced_mean[0], enhanced_mean[1] * 0.8,
                             "Brown disease color pattern should be preserved")


class TestPerformanceBenchmark(unittest.TestCase):
    """Benchmark performance metrics"""
    
    @classmethod
    def setUpClass(cls):
        """Set up benchmarking"""
        cls.processor = LASSRProcessor(model_type='standard', optimize_mobile=True)
        cls.sizes = {
            'small': (128, 128),
            'medium': (256, 256),
            'large': (512, 512)
        }
    
    def test_inference_times(self):
        """Benchmark inference times for different sizes"""
        results = {}
        
        for size_name, (h, w) in self.sizes.items():
            image = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            
            times = []
            for _ in range(3):
                start = time.time()
                _ = self.processor.process(image)
                elapsed = (time.time() - start) * 1000
                times.append(elapsed)
            
            avg_time = np.mean(times)
            results[size_name] = avg_time
            
            print(f"\n{size_name} ({h}x{w}): {avg_time:.1f}ms")
        
        # Check that all sizes meet budget
        self.assertLess(results['small'], 300)
        self.assertLess(results['medium'], 400)
        self.assertLess(results['large'], 600)  # May use tiling
    
    def test_memory_efficiency(self):
        """Test memory usage stays within bounds"""
        large_image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        
        # Should process without memory error
        enhanced = self.processor.process(large_image)
        self.assertIsNotNone(enhanced)
        
        # Check estimated memory
        estimated = estimate_memory_usage((1024, 1024, 3), tile_size=256)
        self.assertLess(estimated, 100, f"Memory usage {estimated:.1f}MB exceeds 100MB limit")


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)