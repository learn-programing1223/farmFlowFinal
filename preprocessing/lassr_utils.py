"""
Utility functions for LASSR processing
Handles image preprocessing, tiling, and memory management
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Optional
import warnings


def preprocess_image(image: np.ndarray, 
                    device: torch.device,
                    normalize: bool = True) -> torch.Tensor:
    """
    Preprocess image for LASSR model
    
    Args:
        image: Input image (H, W, C) in RGB format, uint8
        device: Target device for tensor
        normalize: Whether to normalize to [0, 1]
    
    Returns:
        Preprocessed tensor (1, C, H, W)
    """
    # Ensure RGB format
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = image[:, :, :3]  # Remove alpha channel
    
    # Convert to float and normalize
    if image.dtype == np.uint8:
        image = image.astype(np.float32)
        if normalize:
            image = image / 255.0
    
    # Convert to tensor and add batch dimension
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    
    return tensor.to(device)


def postprocess_image(tensor: torch.Tensor,
                     denormalize: bool = True) -> np.ndarray:
    """
    Convert model output tensor back to image
    
    Args:
        tensor: Output tensor (1, C, H, W)
        denormalize: Whether to denormalize from [0, 1] to [0, 255]
    
    Returns:
        Image array (H, W, C) in RGB format, uint8
    """
    # Remove batch dimension and move to CPU
    tensor = tensor.squeeze(0).cpu()
    
    # Clamp values
    tensor = torch.clamp(tensor, 0, 1 if denormalize else 255)
    
    # Convert to numpy
    image = tensor.permute(1, 2, 0).numpy()
    
    # Denormalize if needed
    if denormalize:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    
    return image


def tile_process(image: np.ndarray,
                model: torch.nn.Module,
                tile_size: int = 256,
                overlap: int = 32,
                device: torch.device = torch.device('cpu')) -> np.ndarray:
    """
    Process large image in tiles to manage memory
    
    Args:
        image: Input image
        model: LASSR model
        tile_size: Size of each tile
        overlap: Overlap between tiles for seamless blending
        device: Processing device
    
    Returns:
        Enhanced image
    """
    h, w = image.shape[:2]
    scale = 2  # LASSR uses 2x upscaling
    
    # Calculate output size
    out_h, out_w = h * scale, w * scale
    output = np.zeros((out_h, out_w, 3), dtype=np.float32)
    weight_map = np.zeros((out_h, out_w, 1), dtype=np.float32)
    
    # Calculate tile positions
    tiles = []
    for y in range(0, h, tile_size - overlap):
        for x in range(0, w, tile_size - overlap):
            x_end = min(x + tile_size, w)
            y_end = min(y + tile_size, h)
            
            # Adjust start position for edge tiles
            if x_end == w and x_end - x < tile_size:
                x = max(0, w - tile_size)
            if y_end == h and y_end - y < tile_size:
                y = max(0, h - tile_size)
            
            tiles.append((x, y, min(x + tile_size, w), min(y + tile_size, h)))
    
    # Process each tile
    for x_start, y_start, x_end, y_end in tiles:
        # Extract tile
        tile = image[y_start:y_end, x_start:x_end]
        
        # Pad if necessary
        pad_h = tile_size - tile.shape[0]
        pad_w = tile_size - tile.shape[1]
        if pad_h > 0 or pad_w > 0:
            tile = np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        
        # Process tile
        tile_tensor = preprocess_image(tile, device)
        with torch.no_grad():
            enhanced_tile_tensor = model(tile_tensor)
        enhanced_tile = postprocess_image(enhanced_tile_tensor, denormalize=False)
        
        # Remove padding if applied
        if pad_h > 0 or pad_w > 0:
            enhanced_tile = enhanced_tile[:enhanced_tile.shape[0]-pad_h*scale, 
                                        :enhanced_tile.shape[1]-pad_w*scale]
        
        # Calculate output position
        out_x_start = x_start * scale
        out_y_start = y_start * scale
        out_x_end = min(x_end * scale, out_w)
        out_y_end = min(y_end * scale, out_h)
        
        # Create weight mask for blending (higher weight in center)
        tile_h, tile_w = enhanced_tile.shape[:2]
        weight = create_weight_mask(tile_h, tile_w, overlap * scale)
        
        # Add to output with blending
        output[out_y_start:out_y_end, out_x_start:out_x_end] += enhanced_tile * weight
        weight_map[out_y_start:out_y_end, out_x_start:out_x_end] += weight
    
    # Normalize by weight map
    output = output / np.maximum(weight_map, 1e-8)
    
    return (output * 255).astype(np.uint8)


def create_weight_mask(h: int, w: int, border: int) -> np.ndarray:
    """
    Create weight mask for tile blending
    Higher weights in center, lower at borders for smooth blending
    """
    mask = np.ones((h, w, 1), dtype=np.float32)
    
    if border > 0:
        # Create gradient at borders
        for i in range(border):
            weight = (i + 1) / border
            
            # Top and bottom borders
            if i < h:
                mask[i, :] *= weight
                mask[h - 1 - i, :] *= weight
            
            # Left and right borders
            if i < w:
                mask[:, i] *= weight
                mask[:, w - 1 - i] *= weight
    
    return mask


def estimate_memory_usage(image_shape: Tuple[int, int, int],
                         tile_size: int = 256) -> float:
    """
    Estimate memory usage for processing
    
    Args:
        image_shape: (H, W, C) shape of input image
        tile_size: Size of tiles if tiling is used
    
    Returns:
        Estimated memory usage in MB
    """
    h, w, c = image_shape
    
    # If image fits in single tile
    if h <= tile_size and w <= tile_size:
        # Input + output + model intermediate
        pixels = h * w * c
        memory_bytes = pixels * 4 * 3  # float32, 3x for input/output/intermediate
    else:
        # Tile-based processing
        pixels = tile_size * tile_size * c
        memory_bytes = pixels * 4 * 3
    
    # Add model parameters (roughly 10MB for our architecture)
    memory_bytes += 10 * 1024 * 1024
    
    return memory_bytes / (1024 * 1024)  # Convert to MB


def detect_disease_regions(image: np.ndarray) -> np.ndarray:
    """
    Detect potential disease regions for focused enhancement
    Returns a mask highlighting disease-likely areas
    
    Args:
        image: Input image in RGB format
    
    Returns:
        Binary mask (0-1) highlighting disease regions
    """
    # Convert to HSV for better color-based detection
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Detect common disease indicators
    masks = []
    
    # Brown spots (blight)
    lower_brown = np.array([10, 50, 50])
    upper_brown = np.array([20, 255, 200])
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
    masks.append(brown_mask)
    
    # Yellow areas (nutrient deficiency, mosaic virus)
    lower_yellow = np.array([20, 50, 50])
    upper_yellow = np.array([30, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    masks.append(yellow_mask)
    
    # White/gray areas (powdery mildew)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    masks.append(white_mask)
    
    # Dark spots (various diseases)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, dark_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    masks.append(dark_mask)
    
    # Combine all masks
    combined = np.zeros_like(masks[0], dtype=np.float32)
    for mask in masks:
        combined = np.maximum(combined, mask.astype(np.float32) / 255.0)
    
    # Smooth the mask
    combined = cv2.GaussianBlur(combined, (5, 5), 0)
    
    # Dilate to include surrounding areas
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.dilate(combined, kernel, iterations=1)
    
    return combined


def apply_clahe(image: np.ndarray, clip_limit: float = 3.0) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    Useful for enhancing disease visibility in poor lighting
    
    Args:
        image: Input image
        clip_limit: Threshold for contrast limiting
    
    Returns:
        Enhanced image
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    
    # Convert back to RGB
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    return enhanced


def remove_noise(image: np.ndarray, 
                preserve_edges: bool = True) -> np.ndarray:
    """
    Remove noise while preserving disease patterns
    
    Args:
        image: Input image
        preserve_edges: Whether to preserve edges (important for lesions)
    
    Returns:
        Denoised image
    """
    if preserve_edges:
        # Use bilateral filter to preserve edges
        denoised = cv2.bilateralFilter(image, 5, 50, 50)
    else:
        # Use Gaussian blur for general denoising
        denoised = cv2.GaussianBlur(image, (3, 3), 0)
    
    return denoised


def correct_exposure(image: np.ndarray) -> np.ndarray:
    """
    Correct exposure issues common in field photography
    
    Args:
        image: Input image
    
    Returns:
        Exposure-corrected image
    """
    # Calculate mean brightness
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mean_brightness = np.mean(gray)
    
    # Target brightness (optimal for disease detection)
    target_brightness = 127
    
    # Calculate correction factor
    if mean_brightness > 0:
        factor = target_brightness / mean_brightness
        factor = np.clip(factor, 0.5, 2.0)  # Limit correction
        
        # Apply correction
        corrected = np.clip(image.astype(np.float32) * factor, 0, 255)
        return corrected.astype(np.uint8)
    
    return image


def validate_image(image: np.ndarray) -> Tuple[bool, str]:
    """
    Validate if image is suitable for processing
    
    Args:
        image: Input image
    
    Returns:
        Tuple of (is_valid, message)
    """
    # Check shape
    if len(image.shape) != 3:
        return False, "Image must be 3-dimensional (H, W, C)"
    
    h, w, c = image.shape
    
    # Check channels
    if c not in [3, 4]:
        return False, f"Image must have 3 or 4 channels, got {c}"
    
    # Check size
    if h < 32 or w < 32:
        return False, f"Image too small ({h}x{w}), minimum 32x32"
    
    if h > 4096 or w > 4096:
        return False, f"Image too large ({h}x{w}), maximum 4096x4096"
    
    # Check if image is not completely black or white
    if np.all(image == 0) or np.all(image == 255):
        return False, "Image is completely black or white"
    
    return True, "Valid"