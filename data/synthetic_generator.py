"""
Synthetic Validation Data Generator
Creates test images for each disease category using pattern generation
Enables testing without real training data
"""

import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional
import os
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from preprocessing.illumination.disease_pattern_generator import DiseasePatternGenerator
import logging

logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """
    Generate synthetic disease images for validation and testing
    Uses the disease pattern generator to create realistic patterns
    """
    
    def __init__(self, output_dir: str = 'data/synthetic_validation'):
        """
        Initialize synthetic data generator
        
        Args:
            output_dir: Directory to save generated images
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize pattern generator
        self.pattern_gen = DiseasePatternGenerator()
        
        # Disease categories (excluding Unknown)
        self.diseases = [
            'healthy',
            'blight',
            'leaf_spot',
            'powdery_mildew',
            'mosaic_virus',
            'nutrient_deficiency'
        ]
        
        # Severity levels for variation
        self.severities = ['mild', 'moderate', 'severe']
        
        # Background variations
        self.backgrounds = ['simple', 'complex', 'field']
        
        # Stats tracking
        self.generated_counts = {disease: 0 for disease in self.diseases}
    
    def generate_disease_batch(self, disease: str, 
                               num_images: int = 20,
                               size: Tuple[int, int] = (384, 384)) -> List[np.ndarray]:
        """
        Generate batch of images for a specific disease
        
        Args:
            disease: Disease type
            num_images: Number of images to generate
            size: Image size
            
        Returns:
            List of generated images
        """
        images = []
        
        for i in range(num_images):
            # Vary severity
            severity = self.severities[i % len(self.severities)]
            
            # Create base healthy leaf
            base = self.pattern_gen.create_healthy_leaf(size)
            
            # Apply disease pattern
            if disease == 'healthy':
                # Just use the healthy leaf
                result = base
            elif disease == 'blight':
                result = self.pattern_gen.create_blight_pattern(base, severity=severity)
            elif disease == 'leaf_spot':
                result = self.pattern_gen.create_leaf_spot_pattern(base, severity=severity)
            elif disease == 'powdery_mildew':
                result = self.pattern_gen.create_powdery_mildew_pattern(base, severity=severity)
            elif disease == 'mosaic_virus':
                result = self.pattern_gen.create_mosaic_virus_pattern(base, severity=severity)
            elif disease == 'nutrient_deficiency':
                result = self.create_nutrient_deficiency(base, severity=severity)
            else:
                logger.warning(f"Unknown disease type: {disease}")
                continue
            
            # Get image array
            if hasattr(result, 'image'):
                image = result.image
            else:
                image = result
            
            # Add background variation
            background_type = self.backgrounds[i % len(self.backgrounds)]
            image = self.add_background(image, background_type)
            
            # Add realistic variations
            image = self.add_variations(image, i)
            
            images.append(image)
            
        self.generated_counts[disease] += len(images)
        logger.info(f"Generated {len(images)} {disease} images")
        
        return images
    
    def create_nutrient_deficiency(self, base_image: np.ndarray,
                                   severity: str = 'moderate') -> np.ndarray:
        """
        Create nutrient deficiency pattern (yellowing, pale leaves)
        
        Args:
            base_image: Base healthy leaf
            severity: Deficiency severity
            
        Returns:
            Image with nutrient deficiency
        """
        image = base_image.copy() if isinstance(base_image, np.ndarray) else base_image
        
        # Convert to float
        if hasattr(image, 'astype'):
            img_float = image.astype(np.float32)
        else:
            img_float = image
        
        # Create yellowing effect
        if severity == 'mild':
            yellow_shift = 0.1
        elif severity == 'moderate':
            yellow_shift = 0.2
        else:  # severe
            yellow_shift = 0.3
        
        # Reduce green, increase yellow
        img_float[:, :, 1] *= (1 - yellow_shift * 0.5)  # Reduce green
        img_float[:, :, 0] += yellow_shift * 50  # Add red
        img_float[:, :, 1] += yellow_shift * 30  # Add some green back
        
        # Create interveinal chlorosis pattern
        h, w = img_float.shape[:2]
        
        # Create vein pattern
        vein_mask = np.zeros((h, w), dtype=np.float32)
        
        # Draw main veins
        cv2.line(vein_mask, (w//2, 0), (w//2, h), 1.0, thickness=3)
        cv2.line(vein_mask, (0, h//2), (w, h//2), 1.0, thickness=3)
        
        # Add secondary veins
        for i in range(4):
            angle = i * 45
            x1 = int(w//2 + 100 * np.cos(np.radians(angle)))
            y1 = int(h//2 + 100 * np.sin(np.radians(angle)))
            cv2.line(vein_mask, (w//2, h//2), (x1, y1), 0.7, thickness=2)
        
        # Blur vein mask
        vein_mask = cv2.GaussianBlur(vein_mask, (15, 15), 0)
        
        # Apply chlorosis (yellowing between veins)
        chlorosis_mask = 1 - vein_mask
        for c in range(3):
            img_float[:, :, c] = img_float[:, :, c] * (1 - chlorosis_mask * yellow_shift)
        
        # Add yellow to chlorotic areas
        img_float[:, :, 0] += chlorosis_mask * yellow_shift * 30  # Red
        img_float[:, :, 1] += chlorosis_mask * yellow_shift * 40  # Green
        
        # Clip values
        img_float = np.clip(img_float, 0, 255)
        
        return img_float.astype(np.uint8)
    
    def add_background(self, image: np.ndarray, 
                      background_type: str) -> np.ndarray:
        """
        Add background to image
        
        Args:
            image: Foreground image
            background_type: Type of background
            
        Returns:
            Image with background
        """
        h, w = image.shape[:2]
        
        if background_type == 'simple':
            # Solid color background
            background = np.full((h, w, 3), [200, 220, 240], dtype=np.uint8)
            
        elif background_type == 'complex':
            # Textured background
            background = np.random.randint(100, 200, (h, w, 3), dtype=np.uint8)
            background = cv2.GaussianBlur(background, (21, 21), 0)
            
        else:  # field
            # Realistic field background (brown/green)
            background = np.zeros((h, w, 3), dtype=np.uint8)
            background[:, :, 0] = np.random.randint(80, 120, (h, w))  # Blue
            background[:, :, 1] = np.random.randint(100, 150, (h, w))  # Green
            background[:, :, 2] = np.random.randint(60, 100, (h, w))  # Red
            background = cv2.GaussianBlur(background, (15, 15), 0)
        
        # Create mask for blending
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
        
        # Blend foreground and background
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) / 255.0
        blended = image * mask_3ch + background * (1 - mask_3ch)
        
        return blended.astype(np.uint8)
    
    def add_variations(self, image: np.ndarray, seed: int) -> np.ndarray:
        """
        Add realistic variations to image
        
        Args:
            image: Input image
            seed: Random seed for variations
            
        Returns:
            Image with variations
        """
        np.random.seed(seed)
        
        # Random rotation
        angle = np.random.uniform(-15, 15)
        center = (image.shape[1]//2, image.shape[0]//2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        
        # Random brightness
        brightness = np.random.uniform(0.8, 1.2)
        image = cv2.convertScaleAbs(image, alpha=brightness, beta=0)
        
        # Random blur (simulate focus issues)
        if np.random.random() > 0.7:
            kernel_size = np.random.choice([3, 5])
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
        # Random noise
        if np.random.random() > 0.8:
            noise = np.random.randn(*image.shape) * 10
            image = np.clip(image + noise, 0, 255).astype(np.uint8)
        
        return image
    
    def generate_edge_cases(self, num_images: int = 10) -> List[np.ndarray]:
        """
        Generate edge case images for Unknown detection
        
        Args:
            num_images: Number of edge cases
            
        Returns:
            List of edge case images
        """
        edge_cases = []
        
        for i in range(num_images):
            base = self.pattern_gen.create_healthy_leaf((384, 384))
            
            if i % 5 == 0:
                # Multiple diseases
                result = self.pattern_gen.create_blight_pattern(base, severity='mild')
                result = self.pattern_gen.create_powdery_mildew_pattern(
                    result.image if hasattr(result, 'image') else result,
                    severity='mild'
                )
                
            elif i % 5 == 1:
                # Very severe disease
                result = self.pattern_gen.create_blight_pattern(base, severity='severe')
                
            elif i % 5 == 2:
                # Ambiguous pattern
                result = self.create_ambiguous_pattern(base)
                
            elif i % 5 == 3:
                # Non-plant object
                result = self.create_non_plant_image((384, 384))
                
            else:
                # Very low quality
                result = self.pattern_gen.create_leaf_spot_pattern(base)
                if hasattr(result, 'image'):
                    result = result.image
                # Heavy blur and noise
                result = cv2.GaussianBlur(result, (21, 21), 0)
                noise = np.random.randn(*result.shape) * 50
                result = np.clip(result + noise, 0, 255).astype(np.uint8)
            
            if hasattr(result, 'image'):
                image = result.image
            else:
                image = result
                
            edge_cases.append(image)
        
        logger.info(f"Generated {len(edge_cases)} edge case images")
        return edge_cases
    
    def create_ambiguous_pattern(self, base_image: np.ndarray) -> np.ndarray:
        """Create ambiguous disease pattern"""
        # Mix of symptoms that don't clearly match one disease
        image = base_image.copy() if isinstance(base_image, np.ndarray) else base_image
        
        # Add some brown spots
        for _ in range(5):
            x, y = np.random.randint(50, 334, 2)
            cv2.circle(image, (x, y), np.random.randint(5, 15), 
                      (np.random.randint(100, 150), np.random.randint(50, 100), 50), -1)
        
        # Add some white patches
        for _ in range(3):
            x, y = np.random.randint(50, 334, 2)
            cv2.circle(image, (x, y), np.random.randint(10, 20),
                      (200, 200, 200), -1)
        
        # Add yellowing
        image[:, :, 1] *= 0.8
        image[:, :, 0] += 20
        
        return np.clip(image, 0, 255).astype(np.uint8)
    
    def create_non_plant_image(self, size: Tuple[int, int]) -> np.ndarray:
        """Create non-plant image for Unknown detection"""
        h, w = size
        
        # Random geometric shapes
        image = np.full((h, w, 3), 200, dtype=np.uint8)
        
        # Add random shapes
        for _ in range(10):
            shape_type = np.random.choice(['circle', 'rectangle', 'line'])
            color = tuple(np.random.randint(0, 255, 3).tolist())
            
            if shape_type == 'circle':
                center = (np.random.randint(0, w), np.random.randint(0, h))
                radius = np.random.randint(10, 50)
                cv2.circle(image, center, radius, color, -1)
                
            elif shape_type == 'rectangle':
                pt1 = (np.random.randint(0, w), np.random.randint(0, h))
                pt2 = (np.random.randint(0, w), np.random.randint(0, h))
                cv2.rectangle(image, pt1, pt2, color, -1)
                
            else:  # line
                pt1 = (np.random.randint(0, w), np.random.randint(0, h))
                pt2 = (np.random.randint(0, w), np.random.randint(0, h))
                cv2.line(image, pt1, pt2, color, np.random.randint(1, 5))
        
        return image
    
    def generate_full_dataset(self, images_per_class: int = 20,
                             include_edge_cases: bool = True) -> Dict[str, List[np.ndarray]]:
        """
        Generate complete synthetic validation dataset
        
        Args:
            images_per_class: Number of images per disease class
            include_edge_cases: Include edge cases for Unknown detection
            
        Returns:
            Dictionary mapping disease names to image lists
        """
        dataset = {}
        
        # Generate images for each disease
        for disease in self.diseases:
            logger.info(f"Generating {disease} images...")
            dataset[disease] = self.generate_disease_batch(disease, images_per_class)
        
        # Add edge cases
        if include_edge_cases:
            logger.info("Generating edge cases...")
            dataset['edge_cases'] = self.generate_edge_cases(images_per_class // 2)
        
        # Summary
        total_images = sum(len(images) for images in dataset.values())
        logger.info(f"Generated total of {total_images} synthetic images")
        
        for disease, images in dataset.items():
            logger.info(f"  {disease}: {len(images)} images")
        
        return dataset
    
    def save_dataset(self, dataset: Dict[str, List[np.ndarray]]):
        """
        Save dataset to disk
        
        Args:
            dataset: Dictionary of disease -> image lists
        """
        for disease, images in dataset.items():
            disease_dir = self.output_dir / disease
            disease_dir.mkdir(exist_ok=True)
            
            for i, image in enumerate(images):
                filename = disease_dir / f"{disease}_{i:04d}.jpg"
                cv2.imwrite(str(filename), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
            logger.info(f"Saved {len(images)} {disease} images to {disease_dir}")
    
    def create_labeled_dataset(self, images_per_class: int = 20) -> List[Tuple[np.ndarray, int]]:
        """
        Create labeled dataset for validation
        
        Args:
            images_per_class: Images per class
            
        Returns:
            List of (image, label) tuples
        """
        labeled_data = []
        
        for label_idx, disease in enumerate(self.diseases):
            images = self.generate_disease_batch(disease, images_per_class)
            for image in images:
                labeled_data.append((image, label_idx))
        
        # Shuffle
        np.random.shuffle(labeled_data)
        
        return labeled_data


def generate_test_batch():
    """Quick function to generate a test batch"""
    generator = SyntheticDataGenerator()
    
    # Generate small batch for testing
    test_images = {
        'healthy': generator.generate_disease_batch('healthy', 5),
        'blight': generator.generate_disease_batch('blight', 5),
        'leaf_spot': generator.generate_disease_batch('leaf_spot', 5),
        'edge_cases': generator.generate_edge_cases(5)
    }
    
    logger.info("Test batch generated successfully")
    return test_images


if __name__ == "__main__":
    # Generate synthetic validation dataset
    generator = SyntheticDataGenerator()
    
    print("Generating synthetic validation dataset...")
    dataset = generator.generate_full_dataset(
        images_per_class=20,
        include_edge_cases=True
    )
    
    print("\nSaving dataset to disk...")
    generator.save_dataset(dataset)
    
    print("\nDataset generation complete!")
    print(f"Images saved to: {generator.output_dir}")
    
    # Create labeled dataset for validation
    print("\nCreating labeled dataset...")
    labeled_data = generator.create_labeled_dataset(10)
    print(f"Created {len(labeled_data)} labeled samples")