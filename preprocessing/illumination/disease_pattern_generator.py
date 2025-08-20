"""
Synthetic Disease Pattern Generator for Testing Illumination Normalization
Creates realistic disease patterns to validate preservation through preprocessing
"""

import numpy as np
import cv2
from typing import Tuple, List, Optional
import random
from dataclasses import dataclass


@dataclass
class DiseasePattern:
    """Container for disease pattern with metadata"""
    image: np.ndarray
    mask: np.ndarray
    disease_type: str
    severity: str  # mild, moderate, severe


class DiseasePatternGenerator:
    """
    Generate synthetic disease patterns for testing
    Focuses on visual patterns that appear across multiple plant species
    """
    
    def __init__(self, base_size: Tuple[int, int] = (512, 512)):
        """
        Initialize generator
        
        Args:
            base_size: Default size for generated images
        """
        self.base_size = base_size
        random.seed(42)  # For reproducibility
        np.random.seed(42)
    
    def create_healthy_leaf(self, 
                          size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Create a base healthy leaf image
        
        Args:
            size: Image size (H, W)
        
        Returns:
            Healthy green leaf image
        """
        if size is None:
            size = self.base_size
        
        h, w = size
        
        # Create base green leaf
        image = np.ones((h, w, 3), dtype=np.uint8)
        
        # Healthy green color with slight variations
        base_green = np.array([60, 120, 60])  # RGB
        
        # Add natural color variation
        noise = np.random.normal(0, 10, (h, w, 3))
        image = image * base_green + noise
        
        # Add leaf texture using Perlin-like noise
        texture = self._generate_leaf_texture(size)
        image = image * (0.8 + 0.2 * texture[:, :, np.newaxis])
        
        # Add veins
        image = self._add_leaf_veins(image)
        
        # Clip and convert to uint8
        image = np.clip(image, 0, 255).astype(np.uint8)
        
        return image
    
    def create_blight_pattern(self, 
                            base_image: Optional[np.ndarray] = None,
                            severity: str = 'moderate') -> DiseasePattern:
        """
        Create blight pattern: dark, spreading lesions with irregular borders
        
        Args:
            base_image: Base leaf image (or creates new)
            severity: 'mild', 'moderate', or 'severe'
        
        Returns:
            DiseasePattern with blight symptoms
        """
        if base_image is None:
            base_image = self.create_healthy_leaf()
        
        image = base_image.copy()
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Number of lesions based on severity
        num_lesions = {'mild': 2, 'moderate': 5, 'severe': 10}[severity]
        
        for _ in range(num_lesions):
            # Random position (avoid edges)
            cx = random.randint(w//4, 3*w//4)
            cy = random.randint(h//4, 3*h//4)
            
            # Create irregular lesion shape
            lesion_mask = self._create_irregular_shape(
                (h, w), (cx, cy), 
                radius=random.randint(15, 40)
            )
            
            # Apply dark brown lesion with gradient
            lesion_color_center = np.array([30, 20, 10])  # Very dark brown
            lesion_color_edge = np.array([60, 40, 20])    # Lighter brown
            
            # Distance transform for gradient
            dist = cv2.distanceTransform(lesion_mask, cv2.DIST_L2, 5)
            if dist.max() > 0:
                dist = dist / dist.max()
            
            # Apply gradient coloring
            for c in range(3):
                color_gradient = lesion_color_center[c] + \
                                (lesion_color_edge[c] - lesion_color_center[c]) * dist
                image[:, :, c] = np.where(
                    lesion_mask > 0,
                    color_gradient,
                    image[:, :, c]
                )
            
            # Add necrotic texture
            texture = np.random.uniform(0.7, 1.0, (h, w))
            image = np.where(
                lesion_mask[:, :, np.newaxis] > 0,
                image * texture[:, :, np.newaxis],
                image
            )
            
            # Update mask
            mask = cv2.bitwise_or(mask, lesion_mask)
        
        # Add yellow halo around lesions (characteristic of blight)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        halo = cv2.dilate(mask, kernel) - mask
        
        yellow_tint = np.array([150, 150, 50])
        for c in range(3):
            image[:, :, c] = np.where(
                halo > 0,
                image[:, :, c] * 0.7 + yellow_tint[c] * 0.3,
                image[:, :, c]
            )
        
        image = np.clip(image, 0, 255).astype(np.uint8)
        
        return DiseasePattern(image, mask, 'blight', severity)
    
    def create_leaf_spot_pattern(self,
                                base_image: Optional[np.ndarray] = None,
                                severity: str = 'moderate') -> DiseasePattern:
        """
        Create leaf spot pattern: circular spots with defined borders
        
        Args:
            base_image: Base leaf image
            severity: 'mild', 'moderate', or 'severe'
        
        Returns:
            DiseasePattern with leaf spot symptoms
        """
        if base_image is None:
            base_image = self.create_healthy_leaf()
        
        image = base_image.copy()
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Number of spots based on severity
        num_spots = {'mild': 5, 'moderate': 15, 'severe': 30}[severity]
        
        for _ in range(num_spots):
            # Random position
            cx = random.randint(20, w - 20)
            cy = random.randint(20, h - 20)
            
            # Spot size
            radius = random.randint(5, 15)
            
            # Create concentric circles (characteristic of leaf spot)
            # Outer ring
            cv2.circle(mask, (cx, cy), radius, 255, -1)
            
            # Draw concentric rings with different colors
            colors = [
                (100, 70, 40),   # Outer ring - medium brown
                (80, 50, 30),    # Middle ring - darker
                (60, 30, 20),    # Inner ring - darkest
            ]
            
            for i, (r_factor, color) in enumerate(zip([1.0, 0.7, 0.4], colors)):
                r = int(radius * r_factor)
                if r > 0:
                    spot_mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.circle(spot_mask, (cx, cy), r, 255, -1)
                    
                    # Apply color
                    for c in range(3):
                        image[:, :, c] = np.where(
                            spot_mask > 0,
                            color[c],
                            image[:, :, c]
                        )
            
            # Add dark border
            border = cv2.circle(np.zeros((h, w), dtype=np.uint8), 
                              (cx, cy), radius, 255, 2)
            for c in range(3):
                image[:, :, c] = np.where(
                    border > 0,
                    20,  # Very dark border
                    image[:, :, c]
                )
        
        image = np.clip(image, 0, 255).astype(np.uint8)
        
        return DiseasePattern(image, mask, 'leaf_spot', severity)
    
    def create_powdery_mildew_pattern(self,
                                     base_image: Optional[np.ndarray] = None,
                                     severity: str = 'moderate') -> DiseasePattern:
        """
        Create powdery mildew pattern: white/gray powdery coating
        
        Args:
            base_image: Base leaf image
            severity: 'mild', 'moderate', or 'severe'
        
        Returns:
            DiseasePattern with powdery mildew symptoms
        """
        if base_image is None:
            base_image = self.create_healthy_leaf()
        
        image = base_image.copy().astype(np.float32)
        h, w = image.shape[:2]
        
        # Coverage based on severity
        coverage = {'mild': 0.2, 'moderate': 0.5, 'severe': 0.8}[severity]
        
        # Create cloudy/powdery texture using Perlin-like noise
        powder_layer = self._generate_powder_texture((h, w))
        
        # Threshold to create patchy coverage
        threshold = 1.0 - coverage
        powder_mask = (powder_layer > threshold).astype(np.float32)
        
        # Smooth the mask for realistic appearance
        powder_mask = cv2.GaussianBlur(powder_mask, (15, 15), 0)
        
        # Create white/gray powder color
        powder_color = np.array([200, 200, 200])  # Light gray-white
        
        # Add variation to powder color
        color_variation = np.random.uniform(0.9, 1.1, (h, w, 3))
        powder_colored = powder_color * color_variation
        
        # Blend with original image
        alpha = powder_mask[:, :, np.newaxis] * 0.7  # Semi-transparent
        image = (1 - alpha) * image + alpha * powder_colored
        
        # Add some texture to powder areas
        texture = np.random.uniform(0.95, 1.05, (h, w, 1))
        image = np.where(
            powder_mask[:, :, np.newaxis] > 0.1,
            image * texture,
            image
        )
        
        image = np.clip(image, 0, 255).astype(np.uint8)
        mask = (powder_mask * 255).astype(np.uint8)
        
        return DiseasePattern(image, mask, 'powdery_mildew', severity)
    
    def create_mosaic_virus_pattern(self,
                                   base_image: Optional[np.ndarray] = None,
                                   severity: str = 'moderate') -> DiseasePattern:
        """
        Create mosaic virus pattern: mottled yellow/green patterns
        
        Args:
            base_image: Base leaf image
            severity: 'mild', 'moderate', or 'severe'
        
        Returns:
            DiseasePattern with mosaic virus symptoms
        """
        if base_image is None:
            base_image = self.create_healthy_leaf()
        
        image = base_image.copy().astype(np.float32)
        h, w = image.shape[:2]
        
        # Create Voronoi-like regions for mottled appearance
        num_regions = {'mild': 10, 'moderate': 20, 'severe': 40}[severity]
        
        # Generate random points
        points = np.random.rand(num_regions, 2)
        points[:, 0] *= h
        points[:, 1] *= w
        
        # Create regions using distance to nearest point
        y, x = np.ogrid[:h, :w]
        regions = np.zeros((h, w), dtype=np.int32)
        
        for i, (py, px) in enumerate(points):
            dist = np.sqrt((y - py)**2 + (x - px)**2)
            regions = np.where(dist < np.sqrt((h**2 + w**2) / num_regions), i, regions)
        
        # Assign colors to regions (alternating yellow-green and darker green)
        mask = np.zeros((h, w), dtype=np.uint8)
        
        for i in range(num_regions):
            region_mask = (regions == i)
            
            if i % 2 == 0:
                # Yellow-green areas (chlorotic)
                color = np.array([120, 150, 60])  # Yellow-green
                mask[region_mask] = 255
            else:
                # Darker green areas
                color = np.array([40, 80, 40])  # Dark green
            
            # Apply color with smooth transition
            for c in range(3):
                image[:, :, c] = np.where(
                    region_mask,
                    image[:, :, c] * 0.3 + color[c] * 0.7,
                    image[:, :, c]
                )
        
        # Smooth transitions between regions
        image = cv2.bilateralFilter(image.astype(np.uint8), 9, 75, 75)
        
        # Add vein distortion (characteristic of mosaic virus)
        image = self._add_vein_distortion(image)
        
        image = np.clip(image, 0, 255).astype(np.uint8)
        
        return DiseasePattern(image, mask, 'mosaic_virus', severity)
    
    def apply_lighting_condition(self,
                                disease_pattern: DiseasePattern,
                                condition: str) -> np.ndarray:
        """
        Apply various lighting conditions to test Retinex robustness
        
        Args:
            disease_pattern: Disease pattern to modify
            condition: 'overexposed', 'underexposed', 'harsh_shadow', 'backlit'
        
        Returns:
            Image with lighting condition applied
        """
        image = disease_pattern.image.copy().astype(np.float32)
        h, w = image.shape[:2]
        
        if condition == 'overexposed':
            # Simulate harsh sunlight
            image = image * 1.8 + 50
            # Add blown-out areas
            hotspots = self._create_hotspots((h, w), num_spots=3)
            image = image + hotspots[:, :, np.newaxis] * 100
            
        elif condition == 'underexposed':
            # Simulate deep shade
            image = image * 0.3
            # Add noise (common in low light)
            noise = np.random.normal(0, 5, image.shape)
            image = image + noise
            
        elif condition == 'harsh_shadow':
            # Create shadow gradient
            shadow_mask = np.zeros((h, w), dtype=np.float32)
            shadow_mask[:, :w//2] = 0.3  # Left side in shadow
            shadow_mask[:, w//2:] = 1.0  # Right side in light
            
            # Smooth transition
            shadow_mask = cv2.GaussianBlur(shadow_mask, (51, 51), 0)
            
            # Apply shadow
            image = image * shadow_mask[:, :, np.newaxis]
            
        elif condition == 'backlit':
            # Create radial gradient (darker center, bright edges)
            center = (h // 2, w // 2)
            y, x = np.ogrid[:h, :w]
            dist = np.sqrt((x - center[1])**2 + (y - center[0])**2)
            max_dist = np.sqrt(center[0]**2 + center[1]**2)
            
            backlight = 0.3 + 0.7 * (dist / max_dist)
            image = image * backlight[:, :, np.newaxis]
        
        image = np.clip(image, 0, 255).astype(np.uint8)
        
        return image
    
    def _generate_leaf_texture(self, size: Tuple[int, int]) -> np.ndarray:
        """Generate natural leaf texture using simplified Perlin noise"""
        h, w = size
        
        # Create multi-scale noise
        texture = np.zeros((h, w), dtype=np.float32)
        
        for scale in [4, 8, 16, 32]:
            freq = 1.0 / scale
            noise = np.random.randn(h // scale + 1, w // scale + 1)
            noise_resized = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
            texture += noise_resized * freq
        
        # Normalize
        texture = (texture - texture.min()) / (texture.max() - texture.min())
        
        return texture
    
    def _generate_powder_texture(self, size: Tuple[int, int]) -> np.ndarray:
        """Generate powdery/cloudy texture"""
        h, w = size
        
        # Create cloud-like texture
        texture = np.zeros((h, w), dtype=np.float32)
        
        # Multiple octaves for realistic appearance
        for scale in [8, 16, 32, 64]:
            freq = 1.0 / scale
            noise = np.random.randn(h // scale + 1, w // scale + 1)
            noise_smooth = cv2.GaussianBlur(noise, (5, 5), 0)
            noise_resized = cv2.resize(noise_smooth, (w, h), interpolation=cv2.INTER_CUBIC)
            texture += noise_resized * freq
        
        # Normalize and apply sigmoid for cloud-like appearance
        texture = (texture - texture.min()) / (texture.max() - texture.min() + 1e-8)
        texture = 1 / (1 + np.exp(-10 * (texture - 0.5)))  # Sigmoid
        
        return texture
    
    def _add_leaf_veins(self, image: np.ndarray) -> np.ndarray:
        """Add realistic leaf veins"""
        h, w = image.shape[:2]
        
        # Create main vein
        vein_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Main central vein
        cv2.line(vein_mask, (w//2, 0), (w//2, h), 1, 2)
        
        # Side veins
        for i in range(5):
            y_pos = (i + 1) * h // 6
            # Left side veins
            cv2.line(vein_mask, (w//2, y_pos), (0, y_pos - 20), 1, 1)
            # Right side veins  
            cv2.line(vein_mask, (w//2, y_pos), (w, y_pos - 20), 1, 1)
        
        # Make veins slightly darker
        image = image.astype(np.float32)
        image = np.where(vein_mask[:, :, np.newaxis] > 0,
                        image * 0.85,
                        image)
        
        return image.astype(np.uint8)
    
    def _add_vein_distortion(self, image: np.ndarray) -> np.ndarray:
        """Add vein distortion characteristic of mosaic virus"""
        h, w = image.shape[:2]
        
        # Create distortion map
        flow = np.zeros((h, w, 2), dtype=np.float32)
        
        # Add sinusoidal distortion
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        flow[:, :, 0] = 3 * np.sin(2 * np.pi * y / 50)
        flow[:, :, 1] = 3 * np.sin(2 * np.pi * x / 50)
        
        # Apply distortion
        map_x = (x + flow[:, :, 0]).astype(np.float32)
        map_y = (y + flow[:, :, 1]).astype(np.float32)
        
        distorted = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
        
        return distorted
    
    def _create_irregular_shape(self,
                               size: Tuple[int, int],
                               center: Tuple[int, int],
                               radius: int) -> np.ndarray:
        """Create irregular shape for lesions"""
        h, w = size
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Create irregular polygon
        num_points = random.randint(8, 12)
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        
        # Add randomness to radius
        radii = radius + np.random.uniform(-radius * 0.3, radius * 0.3, num_points)
        
        # Generate points
        points = []
        for angle, r in zip(angles, radii):
            x = int(center[0] + r * np.cos(angle))
            y = int(center[1] + r * np.sin(angle))
            points.append([x, y])
        
        points = np.array(points, dtype=np.int32)
        
        # Draw filled polygon
        cv2.fillPoly(mask, [points], 255)
        
        return mask
    
    def _create_hotspots(self, size: Tuple[int, int], num_spots: int = 3) -> np.ndarray:
        """Create bright hotspots for overexposure"""
        h, w = size
        hotspots = np.zeros((h, w), dtype=np.float32)
        
        for _ in range(num_spots):
            cx = random.randint(w//4, 3*w//4)
            cy = random.randint(h//4, 3*h//4)
            radius = random.randint(30, 60)
            
            # Create radial gradient
            y, x = np.ogrid[:h, :w]
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            
            # Gaussian-like falloff
            spot = np.exp(-(dist**2) / (2 * radius**2))
            hotspots = np.maximum(hotspots, spot)
        
        return hotspots