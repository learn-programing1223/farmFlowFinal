"""
DeepLabV3 Segmentation for Complex Backgrounds (Optional)
Uses pretrained DeepLabV3-MobileNetV2 from torch.hub
This component is optional - system works without it
"""

import numpy as np
import cv2
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)

# Try to import PyTorch - if not available, RGB segmentation will be used
try:
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. DeepLab segmentation disabled.")


class DeepLabSegmentation:
    """
    DeepLabV3 segmentation for complex backgrounds
    Uses pretrained model - no training required
    Falls back gracefully if PyTorch not available
    """
    
    # PASCAL VOC class indices that correspond to plants
    PLANT_CLASSES = [
        5,   # bottle (sometimes plant containers)
        8,   # cat (ignore)
        9,   # chair (ignore)
        15,  # person (ignore)
        18,  # potted plant (PRIMARY)
    ]
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialize DeepLab segmentation
        
        Args:
            device: Device for inference ('cpu' or 'cuda')
        """
        self.device = device
        self.model = None
        self.transform = None
        
        if not TORCH_AVAILABLE:
            logger.warning("DeepLab initialization skipped - PyTorch not available")
            return
        
        try:
            # Load pretrained DeepLabV3-MobileNetV2
            logger.info("Loading DeepLabV3-MobileNetV2...")
            self.model = torch.hub.load(
                'pytorch/vision:v0.10.0',
                'deeplabv3_mobilenet_v2',
                pretrained=True
            )
            
            # Set to evaluation mode
            self.model.eval()
            
            # Move to device
            if device == 'cuda' and torch.cuda.is_available():
                self.model = self.model.cuda()
                self.device = 'cuda'
            else:
                self.model = self.model.cpu()
                self.device = 'cpu'
            
            # Define preprocessing
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            logger.info(f"DeepLab loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load DeepLab: {e}")
            self.model = None
    
    def segment(self, image: np.ndarray, 
                disease_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        """
        Perform segmentation using DeepLab
        
        Args:
            image: RGB image (H, W, 3)
            disease_mask: Pre-detected disease regions to preserve
            
        Returns:
            Tuple of (segmentation_mask, info)
        """
        import time
        start_time = time.time()
        
        info = {
            'method': 'deeplab',
            'success': False,
            'time_ms': 0
        }
        
        if self.model is None:
            # Model not available, return empty mask
            logger.warning("DeepLab model not available")
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            info['success'] = False
            return mask, info
        
        try:
            h_orig, w_orig = image.shape[:2]
            
            # Preprocess image
            input_tensor = self.preprocess(image)
            
            # Run inference
            with torch.no_grad():
                if self.device == 'cuda':
                    input_tensor = input_tensor.cuda()
                
                output = self.model(input_tensor)['out']
                
                # Get predictions
                predictions = output.argmax(dim=1)
                
                # Move to CPU and convert to numpy
                predictions = predictions.cpu().numpy()[0]
            
            # Extract plant regions
            plant_mask = self.extract_plant_mask(predictions)
            
            # Resize to original size
            if plant_mask.shape != (h_orig, w_orig):
                plant_mask = cv2.resize(
                    plant_mask,
                    (w_orig, h_orig),
                    interpolation=cv2.INTER_NEAREST
                )
            
            # CRITICAL: Always include disease mask
            if disease_mask is not None:
                # Ensure same size
                if disease_mask.shape != plant_mask.shape:
                    disease_mask = cv2.resize(disease_mask, 
                                            (w_orig, h_orig),
                                            interpolation=cv2.INTER_NEAREST)
                
                # Union with disease mask
                plant_mask = cv2.bitwise_or(plant_mask, disease_mask)
            
            # Refine mask
            plant_mask = self.refine_mask(plant_mask)
            
            # Calculate metrics
            coverage = np.sum(plant_mask > 0) / (h_orig * w_orig)
            info['coverage'] = coverage
            info['success'] = True
            
        except Exception as e:
            logger.error(f"DeepLab segmentation failed: {e}")
            # Return empty mask on failure
            plant_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            info['success'] = False
        
        # Calculate timing
        elapsed_ms = (time.time() - start_time) * 1000
        info['time_ms'] = elapsed_ms
        
        return plant_mask, info
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for DeepLab
        
        Args:
            image: RGB image (H, W, 3)
            
        Returns:
            Preprocessed tensor
        """
        # Resize to standard size for efficiency
        # DeepLab can handle various sizes but smaller is faster
        target_size = 513  # Standard DeepLab size
        
        h, w = image.shape[:2]
        if max(h, w) > target_size:
            # Resize keeping aspect ratio
            scale = target_size / max(h, w)
            new_h = int(h * scale)
            new_w = int(w * scale)
            image_resized = cv2.resize(image, (new_w, new_h))
        else:
            image_resized = image
        
        # Convert to PIL format expected by transform
        # Note: OpenCV uses BGR, we have RGB
        image_pil = image_resized
        
        # Apply transforms
        input_tensor = self.transform(image_pil)
        
        # Add batch dimension
        input_tensor = input_tensor.unsqueeze(0)
        
        return input_tensor
    
    def extract_plant_mask(self, predictions: np.ndarray) -> np.ndarray:
        """
        Extract plant regions from DeepLab predictions
        
        Args:
            predictions: Class predictions from DeepLab
            
        Returns:
            Binary mask of plant regions
        """
        mask = np.zeros_like(predictions, dtype=np.uint8)
        
        # Method 1: Look for potted plant class (most reliable)
        plant_pixels = (predictions == 18)  # Potted plant class
        mask[plant_pixels] = 255
        
        # Method 2: Use color-based verification
        # Sometimes plants are misclassified as other objects
        # We'll also include any green-ish regions
        
        # If very little detected, be more inclusive
        if np.sum(mask) < (mask.size * 0.01):  # Less than 1%
            # Include anything that's not obviously background
            # Background class is 0
            foreground = (predictions != 0)
            mask[foreground] = 255
        
        return mask
    
    def refine_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Refine the segmentation mask
        
        Args:
            mask: Binary mask
            
        Returns:
            Refined mask
        """
        # Remove small components
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
        
        # Close gaps
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        closed = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)
        
        # Fill holes
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(closed)
        cv2.drawContours(filled, contours, -1, 255, -1)
        
        return filled
    
    def segment_with_colorspace(self, image: np.ndarray,
                               predictions: np.ndarray) -> np.ndarray:
        """
        Combine DeepLab predictions with color-based segmentation
        Helps when DeepLab misses plant regions
        
        Args:
            image: Original RGB image
            predictions: DeepLab predictions
            
        Returns:
            Combined mask
        """
        # Get initial mask from predictions
        deeplab_mask = self.extract_plant_mask(predictions)
        
        # Color-based plant detection
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Green vegetation
        lower_green = np.array([25, 30, 30])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Brown vegetation (diseased)
        lower_brown = np.array([10, 30, 30])
        upper_brown = np.array([20, 255, 200])
        brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
        
        # Combine color masks
        color_mask = cv2.bitwise_or(green_mask, brown_mask)
        
        # Combine with DeepLab predictions
        combined = cv2.bitwise_or(deeplab_mask, color_mask)
        
        return combined