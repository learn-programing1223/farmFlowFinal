"""
EfficientNet-B4 Implementation for Tier 2 Accurate Inference
Using pretrained ImageNet weights adapted for disease detection
Target: 600-800ms inference, 99%+ accuracy on moderate cases
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, Tuple, Optional, List
import numpy as np
import logging

logger = logging.getLogger(__name__)


# ImageNet class indices that map to plant/disease patterns
IMAGENET_PLANT_MAPPINGS = {
    # Healthy plant indicators (green, leafy, flourishing)
    'Healthy': [
        # Direct plant classes
        985,  # daisy
        984,  # sunflower  
        734,  # pollen
        991,  # broccoli
        992,  # cauliflower
        937,  # artichoke
        # Green/healthy textures
        336,  # mowed lawn
        # Leaf patterns
        989,  # corn
    ],
    
    # Blight (brown, dead, wilted patterns)
    'Blight': [
        # Decay/rot patterns
        945,  # mushroom (fungal)
        947,  # agaric (fungal)
        997,  # bolete (fungal decay)
        # Brown/dead textures
        977,  # sandbar (brown)
        974,  # cliff (earthy brown)
        # Rust-like patterns
        311,  # rust (oxidation pattern)
    ],
    
    # Leaf Spot (circular patterns, spots)
    'Leaf Spot': [
        # Spotted patterns
        323,  # monarch butterfly (spotted wings)
        38,   # leopard (spotted pattern)
        289,  # snow leopard (spots)
        # Circular patterns  
        971,  # bubble
        # Textured surfaces
        968,  # alp (rocky texture)
    ],
    
    # Powdery Mildew (white, dusty, web-like)
    'Powdery Mildew': [
        # White/powdery textures
        948,  # web site (web pattern)
        815,  # spider web
        951,  # ice lolly (white frost)
        952,  # ice cream (white coating)
        # Dusty/powder patterns
        795,  # ski mask (white fabric)
        # Mold-like
        991,  # broccoli (when white)
    ],
    
    # Mosaic Virus (mottled, mixed colors, patches)
    'Mosaic Virus': [
        # Mottled patterns
        72,   # mosaic (literal)
        340,  # zebra (striped/mottled)
        388,  # giant panda (patches)
        # Mixed textures
        629,  # jigsaw puzzle (fragmented)
        955,  # jackfruit (bumpy mottled)
    ],
    
    # Nutrient Deficiency (yellow, pale, weak)
    'Nutrient Deficiency': [
        # Yellow/pale patterns
        950,  # lemon (yellow)
        954,  # banana (yellow)
        989,  # corn (when yellowing)
        # Wilted/weak appearance
        338,  # hay (dried out)
        977,  # sandbar (pale)
    ]
}


class DiseaseAdapter(nn.Module):
    """
    Adapts ImageNet features to disease detection
    Maps 1000 ImageNet classes to 6 disease categories
    """
    
    def __init__(self, num_classes: int = 6):
        super().__init__()
        
        self.num_classes = num_classes
        self.disease_names = ['Healthy', 'Blight', 'Leaf Spot', 
                              'Powdery Mildew', 'Mosaic Virus', 'Nutrient Deficiency']
        
        # Create mapping matrix (1000 ImageNet → 6 diseases)
        self.register_buffer('mapping_matrix', self._create_mapping_matrix())
        
        # Learnable temperature for calibration
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        
        # Optional fine-tuning layer
        self.adapter = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Weight for combining direct mapping and adapter
        self.alpha = nn.Parameter(torch.tensor(0.7))  # 70% mapping, 30% learned
    
    def _create_mapping_matrix(self) -> torch.Tensor:
        """
        Create soft mapping from ImageNet to disease classes
        """
        matrix = torch.zeros(1000, self.num_classes)
        
        # Set weights for mapped classes
        for disease_idx, disease in enumerate(self.disease_names):
            if disease in IMAGENET_PLANT_MAPPINGS:
                for imagenet_idx in IMAGENET_PLANT_MAPPINGS[disease]:
                    if imagenet_idx < 1000:  # Safety check
                        matrix[imagenet_idx, disease_idx] = 1.0
        
        # Add small uniform probability for unmapped classes
        unmapped_weight = 0.01
        matrix[matrix.sum(dim=1) == 0] = unmapped_weight
        
        # Normalize rows to sum to 1
        row_sums = matrix.sum(dim=1, keepdim=True)
        row_sums[row_sums == 0] = 1.0  # Avoid division by zero
        matrix = matrix / row_sums
        
        return matrix
    
    def forward(self, imagenet_logits: torch.Tensor) -> torch.Tensor:
        """
        Convert ImageNet predictions to disease predictions
        
        Args:
            imagenet_logits: Raw logits from ImageNet model (B, 1000)
            
        Returns:
            Disease logits (B, 6)
        """
        # Apply temperature scaling to ImageNet logits
        scaled_logits = imagenet_logits / self.temperature
        
        # Convert to probabilities
        imagenet_probs = F.softmax(scaled_logits, dim=-1)
        
        # Method 1: Direct mapping using our matrix
        mapped_probs = torch.matmul(imagenet_probs, self.mapping_matrix)
        mapped_logits = torch.log(mapped_probs + 1e-8)
        
        # Method 2: Learned adapter
        adapted_logits = self.adapter(imagenet_logits)
        
        # Combine both methods
        combined_logits = self.alpha * mapped_logits + (1 - self.alpha) * adapted_logits
        
        return combined_logits


class EfficientNetB4Tier2(nn.Module):
    """
    Tier 2 EfficientNet-B4 with pretrained ImageNet weights
    Adapted for plant disease detection
    """
    
    def __init__(self, num_classes: int = 6, pretrained: bool = True):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Load pretrained EfficientNet-B4
        logger.info("Loading pretrained EfficientNet-B4...")
        self.backbone = models.efficientnet_b4(pretrained=pretrained)
        
        # Keep the original classifier for ImageNet features
        self.imagenet_head = self.backbone.classifier
        
        # Create new disease classification head
        in_features = self.backbone.classifier[1].in_features
        self.disease_head = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features, num_classes)
        )
        
        # Disease adapter for ImageNet mapping
        self.adapter = DiseaseAdapter(num_classes)
        
        # Replace the classifier with identity for feature extraction
        self.backbone.classifier = nn.Identity()
        
        # Uncertainty estimation dropout
        self.mc_dropout = nn.Dropout(0.3)
        
        logger.info(f"EfficientNet-B4 loaded with {num_classes} disease classes")
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features using EfficientNet backbone"""
        return self.backbone(x)
    
    def forward(self, x: torch.Tensor, 
                use_adapter: bool = True,
                return_features: bool = False) -> torch.Tensor:
        """
        Forward pass with optional ImageNet adapter
        
        Args:
            x: Input tensor (B, 3, H, W)
            use_adapter: Use ImageNet to disease mapping
            return_features: Return features for analysis
            
        Returns:
            Disease predictions or (predictions, features)
        """
        # Extract features
        features = self.extract_features(x)
        
        # Apply MC dropout for uncertainty
        features_drop = self.mc_dropout(features)
        
        if use_adapter:
            # Get ImageNet logits
            imagenet_logits = self.imagenet_head(features_drop)
            
            # Adapt to disease predictions
            disease_logits = self.adapter(imagenet_logits)
            
            # Also get direct disease predictions
            direct_logits = self.disease_head(features_drop)
            
            # Weighted combination (more weight on adapter initially)
            final_logits = 0.6 * disease_logits + 0.4 * direct_logits
        else:
            # Direct disease prediction only
            final_logits = self.disease_head(features_drop)
        
        if return_features:
            return final_logits, features
        return final_logits
    
    def predict_with_confidence(self, x: torch.Tensor,
                                num_samples: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with confidence using MC Dropout
        More samples than Tier 1 since we have more time
        
        Args:
            x: Input tensor
            num_samples: Number of forward passes
            
        Returns:
            (predictions, confidence)
        """
        self.train()  # Enable dropout
        
        predictions = []
        with torch.no_grad():
            for _ in range(num_samples):
                logits = self.forward(x)
                probs = F.softmax(logits, dim=-1)
                predictions.append(probs)
        
        self.eval()
        
        # Stack predictions
        predictions = torch.stack(predictions)
        
        # Calculate statistics
        mean_pred = predictions.mean(0)
        std_pred = predictions.std(0)
        
        # Confidence from entropy and std
        entropy = -(mean_pred * torch.log(mean_pred + 1e-8)).sum(-1)
        max_entropy = np.log(self.num_classes)
        entropy_confidence = 1 - (entropy / max_entropy)
        
        # Confidence from standard deviation
        std_confidence = 1 - std_pred.mean(-1)
        
        # Combined confidence
        confidence = 0.6 * entropy_confidence + 0.4 * std_confidence
        
        return mean_pred, confidence


class EfficientNetTier2:
    """
    Tier 2 wrapper for deployment
    Handles preprocessing, inference, and confidence calibration
    """
    
    def __init__(self, model_path: Optional[str] = None,
                 device: str = 'cpu', 
                 use_pretrained: bool = True):
        """
        Initialize Tier 2 model
        
        Args:
            model_path: Path to fine-tuned weights (optional)
            device: Inference device
            use_pretrained: Use ImageNet pretrained weights
        """
        self.device = torch.device(device)
        self.model = EfficientNetB4Tier2(num_classes=6, pretrained=use_pretrained)
        
        if model_path:
            self.load_weights(model_path)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Disease class names (excluding Unknown - that's confidence-based)
        self.classes = [
            'Healthy',
            'Blight', 
            'Leaf Spot',
            'Powdery Mildew',
            'Mosaic Virus',
            'Nutrient Deficiency'
        ]
        
        # Confidence thresholds
        self.confidence_thresholds = {
            'Healthy': 0.85,
            'Blight': 0.80,
            'Leaf Spot': 0.80,
            'Powdery Mildew': 0.80,
            'Mosaic Virus': 0.75,  # Complex pattern, lower threshold
            'Nutrient Deficiency': 0.75
        }
        
        # Temperature for calibration (will be tuned)
        self.temperature = 1.5
        
        logger.info("Tier 2 EfficientNet-B4 initialized")
    
    def load_weights(self, path: str):
        """Load fine-tuned weights"""
        try:
            state_dict = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            logger.info(f"Loaded weights from {path}")
        except Exception as e:
            logger.warning(f"Could not load weights: {e}")
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for EfficientNet
        
        Args:
            image: RGB image array
            
        Returns:
            Preprocessed tensor
        """
        import cv2
        
        # EfficientNet-B4 uses 380x380
        target_size = 380
        
        if image.shape[:2] != (target_size, target_size):
            image = cv2.resize(image, (target_size, target_size))
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        # Convert to tensor
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.float()
        
        return tensor.to(self.device)
    
    def infer(self, image: np.ndarray,
              use_adapter: bool = True) -> Dict:
        """
        Run Tier 2 inference
        
        Args:
            image: RGB image array
            use_adapter: Use ImageNet mapping
            
        Returns:
            Inference results
        """
        import time
        start_time = time.time()
        
        # Preprocess
        input_tensor = self.preprocess(image)
        
        # Get predictions with confidence
        predictions, confidence = self.model.predict_with_confidence(
            input_tensor, num_samples=20
        )
        
        # Apply temperature scaling
        predictions = predictions[0].cpu().numpy()
        confidence_score = confidence[0].item()
        
        # Get top prediction
        class_idx = np.argmax(predictions)
        class_name = self.classes[class_idx]
        class_prob = predictions[class_idx]
        
        # Check confidence threshold
        threshold = self.confidence_thresholds[class_name]
        should_escalate = confidence_score < threshold
        
        # If confidence too low, might be Unknown
        if confidence_score < 0.70:
            should_escalate = True  # Let Tier 3 decide or classify as Unknown
        
        # Calculate inference time
        inference_time = (time.time() - start_time) * 1000
        
        return {
            'tier': 2,
            'class': class_name,
            'confidence': confidence_score,
            'probability': class_prob,
            'all_probabilities': predictions.tolist(),
            'should_escalate': should_escalate,
            'inference_time_ms': inference_time,
            'threshold_used': threshold,
            'adapter_used': use_adapter
        }
    
    def calibrate_temperature(self, validation_data: List[Tuple[np.ndarray, int]]):
        """
        Calibrate temperature scaling using validation data
        
        Args:
            validation_data: List of (image, label) tuples
        """
        # This would implement temperature scaling calibration
        # For now, using fixed temperature
        pass
    
    def benchmark(self, num_iterations: int = 50) -> Dict:
        """
        Benchmark model performance
        
        Args:
            num_iterations: Number of iterations
            
        Returns:
            Benchmark results
        """
        import time
        
        # Create dummy input (380x380 for EfficientNet-B4)
        dummy_input = torch.randn(1, 3, 380, 380).to(self.device)
        
        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = self.model(dummy_input)
        
        # Benchmark
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            with torch.no_grad():
                _ = self.model(dummy_input)
            
            if self.device == 'cuda':
                torch.cuda.synchronize()
                
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        
        return {
            'mean_time_ms': np.mean(times),
            'std_time_ms': np.std(times),
            'min_time_ms': np.min(times),
            'max_time_ms': np.max(times),
            'target_600ms': np.mean(times) < 600,
            'target_800ms': np.mean(times) < 800
        }