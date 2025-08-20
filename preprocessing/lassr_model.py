"""
LASSR Model Architecture for Plant Disease Detection
Focus: Enhance disease patterns (lesions, spots, discoloration) across ALL plant species
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class DiseasePatternAttention(nn.Module):
    """
    Attention mechanism specifically designed to focus on disease patterns:
    - Lesions and spots
    - Powdery/fuzzy textures
    - Color abnormalities
    - Pattern irregularities
    """
    
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        
        # Channel attention for disease-specific features
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid()
        )
        
        # Spatial attention for lesion/spot locations
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
        # Texture enhancement for disease patterns
        self.texture_conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        
    def forward(self, x):
        # Channel attention - focus on disease-relevant channels
        ca = self.channel_attention(x)
        x_ca = x * ca
        
        # Spatial attention - focus on disease locations
        avg_pool = torch.mean(x_ca, dim=1, keepdim=True)
        max_pool = torch.max(x_ca, dim=1, keepdim=True)[0]
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        sa = self.spatial_attention(spatial_input)
        x_sa = x_ca * sa
        
        # Texture enhancement for disease patterns
        texture = self.texture_conv(x_sa)
        
        return x + texture  # Residual connection


class ResidualDiseaseBlock(nn.Module):
    """
    Residual block optimized for disease pattern preservation
    """
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.disease_attention = DiseasePatternAttention(channels)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = self.disease_attention(out)
        return self.relu(out + residual)


class LASSRDisease(nn.Module):
    """
    Lightweight Attention-based Super-Resolution for Disease patterns
    Adapted from EDSR for mobile deployment with disease focus
    
    Key features:
    - 2x upscaling for field images
    - Disease pattern preservation
    - < 100MB memory footprint
    - 200-400ms inference time
    """
    
    def __init__(self, 
                 num_channels: int = 3,
                 num_features: int = 64,
                 num_blocks: int = 8,
                 scale: int = 2):
        super().__init__()
        
        self.scale = scale
        
        # Shallow feature extraction
        self.conv_first = nn.Conv2d(num_channels, num_features, 3, padding=1)
        
        # Disease-focused residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualDiseaseBlock(num_features) for _ in range(num_blocks)
        ])
        
        # Feature fusion
        self.conv_middle = nn.Conv2d(num_features, num_features, 3, padding=1)
        
        # Upsampling with pixel shuffle (efficient for mobile)
        self.upscale = nn.Sequential(
            nn.Conv2d(num_features, num_features * (scale ** 2), 3, padding=1),
            nn.PixelShuffle(scale),
            nn.ReLU(inplace=True)
        )
        
        # Final reconstruction focused on disease details
        self.conv_last = nn.Conv2d(num_features, num_channels, 3, padding=1)
        
        # Disease pattern enhancement (learnable)
        self.pattern_enhance = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        
    def forward(self, x):
        # Store original for residual learning
        bicubic = F.interpolate(x, scale_factor=self.scale, mode='bicubic', align_corners=False)
        
        # Shallow features
        x = self.conv_first(x)
        residual = x
        
        # Deep feature extraction with disease attention
        for block in self.residual_blocks:
            x = block(x)
        
        # Feature fusion
        x = self.conv_middle(x)
        x = x + residual  # Global residual
        
        # Upsampling
        x = self.upscale(x)
        
        # Final reconstruction
        x = self.conv_last(x)
        
        # Apply disease pattern enhancement
        x = x * self.pattern_enhance
        
        # Add to bicubic baseline (residual learning)
        return x + bicubic
    
    def load_pretrained_edsr(self, checkpoint_path: Optional[str] = None):
        """
        Load pretrained EDSR weights and adapt for disease detection
        """
        if checkpoint_path:
            # Load pretrained weights
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            
            # Filter out incompatible keys (our model has disease-specific layers)
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() 
                             if k in model_dict and v.shape == model_dict[k].shape}
            
            # Update with pretrained weights
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict, strict=False)
            
            print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} pretrained parameters")
        else:
            # Initialize with Xavier/He initialization for disease patterns
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)


class LightweightLASSR(nn.Module):
    """
    Ultra-lightweight version for fast inference (< 200ms)
    Sacrifices some quality for speed when needed
    """
    
    def __init__(self, num_channels: int = 3, scale: int = 2):
        super().__init__()
        
        # Minimal architecture for speed
        self.conv1 = nn.Conv2d(num_channels, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, num_channels * (scale ** 2), 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        bicubic = F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.pixel_shuffle(x)
        return x + bicubic


def disease_pattern_loss(pred: torch.Tensor, 
                        target: torch.Tensor, 
                        mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Custom loss function prioritizing disease patterns
    
    Args:
        pred: Enhanced image
        target: Ground truth high-res image
        mask: Optional mask highlighting disease regions
    
    Returns:
        Combined loss focusing on disease features
    """
    # Base reconstruction loss
    l1_loss = F.l1_loss(pred, target)
    
    # Edge loss for lesion boundaries
    pred_edges = compute_edges(pred)
    target_edges = compute_edges(target)
    edge_loss = F.l1_loss(pred_edges, target_edges)
    
    # Texture loss using Gram matrices for disease patterns
    texture_loss = gram_matrix_loss(pred, target)
    
    # Color loss for disease-specific discoloration
    color_loss = F.l1_loss(
        torch.mean(pred, dim=(2, 3)),
        torch.mean(target, dim=(2, 3))
    )
    
    # Combine losses with disease-focused weighting
    total_loss = l1_loss + 0.2 * edge_loss + 0.1 * texture_loss + 0.05 * color_loss
    
    # Apply mask if provided (higher weight on disease regions)
    if mask is not None:
        disease_weight = 2.0
        total_loss = mask * total_loss * disease_weight + (1 - mask) * total_loss
    
    return total_loss


def compute_edges(x: torch.Tensor) -> torch.Tensor:
    """
    Compute edges using Sobel filters for boundary preservation
    """
    # Sobel filters
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                           dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                           dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    
    # Apply to each channel
    edges = []
    for i in range(x.shape[1]):
        channel = x[:, i:i+1, :, :]
        edge_x = F.conv2d(channel, sobel_x, padding=1)
        edge_y = F.conv2d(channel, sobel_y, padding=1)
        edge = torch.sqrt(edge_x ** 2 + edge_y ** 2)
        edges.append(edge)
    
    return torch.cat(edges, dim=1)


def gram_matrix_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Gram matrix loss for texture preservation (disease patterns)
    """
    b, c, h, w = pred.shape
    
    # Reshape for matrix multiplication
    pred_flat = pred.view(b, c, h * w)
    target_flat = target.view(b, c, h * w)
    
    # Compute Gram matrices
    gram_pred = torch.bmm(pred_flat, pred_flat.transpose(1, 2)) / (c * h * w)
    gram_target = torch.bmm(target_flat, target_flat.transpose(1, 2)) / (c * h * w)
    
    return F.mse_loss(gram_pred, gram_target)