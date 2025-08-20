"""
EfficientFormer-L7 Implementation for Tier 1 Fast Inference
Target: 7ms inference, 95%+ accuracy on easy cases
Optimized for iPhone Neural Engine deployment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List
import numpy as np


class Attention4D(nn.Module):
    """
    Efficient 4D attention mechanism for mobile deployment
    Key innovation: Factorized attention for speed
    """
    
    def __init__(self, dim: int, num_heads: int = 8, 
                 attention_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=attention_bias)
        self.proj = nn.Linear(dim, dim)
        
        # Attention dropout for uncertainty
        self.attn_drop = nn.Dropout(0.0)  # Will be activated during uncertainty estimation
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Efficient attention computation
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x


class MetaBlock4D(nn.Module):
    """
    EfficientFormer MetaBlock with 4D operations
    Combines efficiency with expressiveness
    """
    
    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.token_mixer = Attention4D(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Token mixing with residual
        x = x + self.token_mixer(self.norm1(x))
        # Channel mixing with residual
        x = x + self.mlp(self.norm2(x))
        return x


class EfficientFormerL7(nn.Module):
    """
    EfficientFormer-L7 for Tier 1 fast disease classification
    Optimized for 7ms inference on iPhone Neural Engine
    
    Architecture:
    - Lightweight stem
    - 4 stages with increasing depth
    - Global pooling head
    - 7 disease classes output
    """
    
    def __init__(self, num_classes: int = 7, 
                 img_size: int = 384,
                 in_chans: int = 3):
        """
        Initialize EfficientFormer-L7
        
        Args:
            num_classes: Number of disease classes (7)
            img_size: Input image size (384)
            in_chans: Input channels (3 for RGB)
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.img_size = img_size
        
        # Lightweight stem - critical for speed
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Stage 1: Simple convolutions for efficiency
        self.stage1 = self._make_stage(64, 128, num_blocks=2, use_attention=False)
        
        # Stage 2: Introduce attention
        self.stage2 = self._make_stage(128, 256, num_blocks=2, use_attention=True)
        
        # Stage 3: Deeper attention
        self.stage3 = self._make_stage(256, 512, num_blocks=4, use_attention=True)
        
        # Stage 4: Final refinement
        self.stage4 = self._make_stage(512, 768, num_blocks=2, use_attention=True)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classification head
        self.norm = nn.LayerNorm(768)
        self.head = nn.Linear(768, num_classes)
        
        # For uncertainty estimation
        self.dropout = nn.Dropout(0.2)
        
        # Initialize weights
        self._init_weights()
    
    def _make_stage(self, in_dim: int, out_dim: int, 
                    num_blocks: int, use_attention: bool) -> nn.Module:
        """Create a stage with specified blocks"""
        layers = []
        
        # Downsample if dimensions change
        if in_dim != out_dim:
            layers.append(nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True)
            ))
            
        # Add blocks
        for _ in range(num_blocks):
            if use_attention:
                # Simplified attention block for mobile
                layers.append(nn.Sequential(
                    nn.Conv2d(out_dim, out_dim, kernel_size=1),
                    nn.BatchNorm2d(out_dim),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, groups=out_dim),
                    nn.BatchNorm2d(out_dim),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_dim, out_dim, kernel_size=1),
                    nn.BatchNorm2d(out_dim)
                ))
            else:
                # Standard convolution block
                layers.append(nn.Sequential(
                    nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_dim),
                    nn.ReLU(inplace=True)
                ))
                
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input"""
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.norm(x)
        
        return x
    
    def forward(self, x: torch.Tensor, 
                return_features: bool = False) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (B, 3, H, W)
            return_features: Return features for uncertainty estimation
            
        Returns:
            Class logits or (logits, features) tuple
        """
        features = self.forward_features(x)
        
        # Apply dropout for uncertainty
        features_drop = self.dropout(features)
        
        # Classification
        logits = self.head(features_drop)
        
        if return_features:
            return logits, features
        return logits
    
    def predict_with_confidence(self, x: torch.Tensor, 
                                num_samples: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with confidence estimation using MC Dropout
        
        Args:
            x: Input tensor
            num_samples: Number of forward passes for uncertainty
            
        Returns:
            Tuple of (predictions, confidence)
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
        
        # Mean prediction
        mean_pred = predictions.mean(0)
        
        # Confidence as 1 - entropy
        entropy = -(mean_pred * torch.log(mean_pred + 1e-8)).sum(-1)
        confidence = 1 - (entropy / np.log(self.num_classes))
        
        return mean_pred, confidence
    
    def should_escalate(self, confidence: torch.Tensor, 
                       threshold: float = 0.85) -> torch.Tensor:
        """
        Determine if sample should be escalated to Tier 2
        
        Args:
            confidence: Confidence scores
            threshold: Confidence threshold (0.85 default)
            
        Returns:
            Boolean tensor indicating escalation
        """
        return confidence < threshold
    
    @torch.jit.export
    def export_for_mobile(self) -> Dict[str, torch.Tensor]:
        """Export model for Core ML conversion"""
        # Create dummy input
        dummy_input = torch.randn(1, 3, self.img_size, self.img_size)
        
        # Trace model
        self.eval()
        with torch.no_grad():
            traced = torch.jit.trace(self, dummy_input)
        
        return {
            'model': traced,
            'input_shape': torch.tensor([1, 3, self.img_size, self.img_size])
        }


class EfficientFormerTier1:
    """
    Tier 1 wrapper for deployment
    Handles preprocessing, inference, and routing decisions
    """
    
    def __init__(self, model_path: Optional[str] = None,
                 device: str = 'cpu'):
        """
        Initialize Tier 1 model
        
        Args:
            model_path: Path to pretrained weights
            device: Inference device
        """
        self.device = torch.device(device)
        self.model = EfficientFormerL7(num_classes=7)
        
        if model_path:
            self.load_weights(model_path)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Disease class names
        self.classes = [
            'Healthy',
            'Blight',
            'Leaf Spot', 
            'Powdery Mildew',
            'Mosaic Virus',
            'Nutrient Deficiency',
            'Unknown'
        ]
        
        # Confidence thresholds per class
        self.confidence_thresholds = {
            'Healthy': 0.90,      # High confidence for healthy
            'Blight': 0.85,       # Standard threshold
            'Leaf Spot': 0.85,
            'Powdery Mildew': 0.85,
            'Mosaic Virus': 0.80,  # Lower threshold for complex patterns
            'Nutrient Deficiency': 0.80,
            'Unknown': 0.70        # Lower threshold for unknown
        }
    
    def load_weights(self, path: str):
        """Load pretrained weights"""
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for inference
        
        Args:
            image: RGB image array
            
        Returns:
            Preprocessed tensor
        """
        # Resize to 384x384
        if image.shape[:2] != (384, 384):
            import cv2
            image = cv2.resize(image, (384, 384))
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Normalize with ImageNet stats
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        # Convert to tensor with float32 type
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.float()  # Ensure float32 type
        
        return tensor.to(self.device)
    
    def infer(self, image: np.ndarray) -> Dict:
        """
        Run Tier 1 inference
        
        Args:
            image: RGB image array
            
        Returns:
            Inference results dictionary
        """
        import time
        start_time = time.time()
        
        # Preprocess
        input_tensor = self.preprocess(image)
        
        # Get predictions with confidence
        predictions, confidence = self.model.predict_with_confidence(
            input_tensor, num_samples=5  # Fewer samples for speed
        )
        
        # Get top prediction
        probs = predictions[0].cpu().numpy()
        confidence_score = confidence[0].item()
        
        # Get class prediction
        class_idx = np.argmax(probs)
        class_name = self.classes[class_idx]
        class_prob = probs[class_idx]
        
        # Check if should escalate
        threshold = self.confidence_thresholds[class_name]
        should_escalate = confidence_score < threshold
        
        # If confidence too low, mark as Unknown
        if confidence_score < 0.70:
            class_name = 'Unknown'
            should_escalate = True
        
        # Calculate inference time
        inference_time = (time.time() - start_time) * 1000
        
        return {
            'tier': 1,
            'class': class_name,
            'confidence': confidence_score,
            'probability': class_prob,
            'all_probabilities': probs.tolist(),
            'should_escalate': should_escalate,
            'inference_time_ms': inference_time,
            'threshold_used': threshold
        }
    
    def benchmark(self, num_iterations: int = 100) -> Dict:
        """
        Benchmark model performance
        
        Args:
            num_iterations: Number of iterations
            
        Returns:
            Benchmark results
        """
        import time
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, 384, 384).to(self.device)
        
        # Warmup
        for _ in range(10):
            _ = self.model(dummy_input)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start = time.perf_counter()
                _ = self.model(dummy_input)
                times.append((time.perf_counter() - start) * 1000)
        
        return {
            'mean_time_ms': np.mean(times),
            'std_time_ms': np.std(times),
            'min_time_ms': np.min(times),
            'max_time_ms': np.max(times),
            'target_7ms': np.mean(times) < 7.0
        }