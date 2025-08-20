"""
HEIC Handler for iPhone Photos
Extracts metadata and handles iPhone-specific image characteristics
"""

import numpy as np
import cv2
from typing import Dict, Optional, Tuple
from pathlib import Path
import logging

# Note: pillow_heif is already in requirements from Day 1
try:
    from PIL import Image
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIF_AVAILABLE = True
except ImportError:
    HEIF_AVAILABLE = False
    
try:
    from PIL.ExifTags import TAGS
    EXIF_AVAILABLE = True
except ImportError:
    EXIF_AVAILABLE = False

logger = logging.getLogger(__name__)


class HEICProcessor:
    """
    Handles iPhone HEIC format photos with metadata extraction
    Uses metadata for adaptive parameter selection
    """
    
    def __init__(self):
        """Initialize HEIC processor"""
        self.supported_extensions = ['.heic', '.heif', '.HEIC', '.HEIF']
        
        # iPhone camera characteristics
        self.iphone_models = {
            'iPhone 12': {'sensor_size': 1/2.55, 'base_iso': 32},
            'iPhone 13': {'sensor_size': 1/1.65, 'base_iso': 32},
            'iPhone 14': {'sensor_size': 1/1.28, 'base_iso': 50},
            'iPhone 15': {'sensor_size': 1/1.28, 'base_iso': 50},
        }
        
    def load_heic(self, file_path: str) -> Tuple[np.ndarray, Dict]:
        """
        Load HEIC file and extract metadata
        
        Args:
            file_path: Path to HEIC file
            
        Returns:
            Tuple of (image_array, metadata_dict)
        """
        if not HEIF_AVAILABLE:
            logger.warning("pillow_heif not available, using fallback")
            return self.fallback_load(file_path)
        
        try:
            # Load with PIL/pillow_heif
            img = Image.open(file_path)
            
            # Extract metadata
            metadata = self.extract_metadata(img)
            
            # Convert to RGB numpy array
            img_rgb = img.convert('RGB')
            img_array = np.array(img_rgb)
            
            # Handle orientation
            img_array = self.correct_orientation(img_array, metadata)
            
            # Color profile conversion if needed
            img_array = self.convert_color_profile(img_array, metadata)
            
            return img_array, metadata
            
        except Exception as e:
            logger.error(f"Error loading HEIC: {e}")
            return self.fallback_load(file_path)
    
    def extract_metadata(self, img: Image.Image) -> Dict:
        """
        Extract relevant metadata from image
        
        Args:
            img: PIL Image object
            
        Returns:
            Dictionary of metadata
        """
        metadata = {
            'format': 'HEIC',
            'has_metadata': False
        }
        
        if not EXIF_AVAILABLE:
            return metadata
        
        try:
            exifdata = img.getexif()
            
            if exifdata:
                metadata['has_metadata'] = True
                
                # Extract key photography parameters
                for tag_id, value in exifdata.items():
                    tag = TAGS.get(tag_id, tag_id)
                    
                    # Key parameters for lighting analysis
                    if tag == 'ExposureTime':
                        metadata['exposure_time'] = value
                    elif tag == 'FNumber':
                        metadata['f_number'] = value
                    elif tag == 'ISOSpeedRatings':
                        metadata['iso'] = value
                    elif tag == 'ExposureBiasValue':
                        metadata['exposure_bias'] = value
                    elif tag == 'WhiteBalance':
                        metadata['white_balance'] = value
                    elif tag == 'Flash':
                        metadata['flash'] = value
                    elif tag == 'LightSource':
                        metadata['light_source'] = value
                    elif tag == 'Make':
                        metadata['make'] = value
                    elif tag == 'Model':
                        metadata['model'] = value
                    elif tag == 'Orientation':
                        metadata['orientation'] = value
                    elif tag == 'DateTimeOriginal':
                        metadata['datetime'] = value
                        
        except Exception as e:
            logger.debug(f"Could not extract EXIF: {e}")
        
        # Analyze metadata for adaptive processing
        metadata['lighting_hints'] = self.analyze_metadata_for_lighting(metadata)
        
        return metadata
    
    def analyze_metadata_for_lighting(self, metadata: Dict) -> Dict:
        """
        Analyze metadata to determine lighting conditions
        
        Args:
            metadata: Extracted metadata
            
        Returns:
            Dictionary of lighting hints
        """
        hints = {
            'likely_outdoor': False,
            'likely_harsh_sun': False,
            'likely_low_light': False,
            'flash_used': False,
            'exposure_compensation': 0
        }
        
        if not metadata.get('has_metadata'):
            return hints
        
        # Check ISO for lighting conditions
        iso = metadata.get('iso', 100)
        if isinstance(iso, (list, tuple)):
            iso = iso[0] if iso else 100
            
        if iso > 800:
            hints['likely_low_light'] = True
        elif iso < 100:
            hints['likely_outdoor'] = True
        
        # Check exposure time
        exposure = metadata.get('exposure_time', 1/60)
        if isinstance(exposure, tuple):
            exposure = exposure[0] / exposure[1] if exposure[1] != 0 else 1/60
            
        # Fast shutter + low ISO = bright conditions
        if exposure < 1/500 and iso < 200:
            hints['likely_harsh_sun'] = True
        
        # Check flash
        flash = metadata.get('flash', 0)
        hints['flash_used'] = bool(flash & 0x1) if isinstance(flash, int) else False
        
        # Exposure compensation
        bias = metadata.get('exposure_bias', 0)
        if isinstance(bias, tuple):
            bias = bias[0] / bias[1] if bias[1] != 0 else 0
        hints['exposure_compensation'] = float(bias)
        
        # Light source (helpful for indoor/outdoor detection)
        light_source = metadata.get('light_source', 0)
        if light_source in [1, 2, 3]:  # Daylight, Fluorescent, Tungsten
            hints['likely_outdoor'] = (light_source == 1)
        
        return hints
    
    def correct_orientation(self, image: np.ndarray, metadata: Dict) -> np.ndarray:
        """
        Correct image orientation based on EXIF data
        
        Args:
            image: Image array
            metadata: Metadata dictionary
            
        Returns:
            Correctly oriented image
        """
        orientation = metadata.get('orientation', 1)
        
        if orientation == 2:  # Horizontal flip
            image = cv2.flip(image, 1)
        elif orientation == 3:  # 180 rotation
            image = cv2.rotate(image, cv2.ROTATE_180)
        elif orientation == 4:  # Vertical flip
            image = cv2.flip(image, 0)
        elif orientation == 5:  # Transpose
            image = cv2.transpose(image)
            image = cv2.flip(image, 0)
        elif orientation == 6:  # 90 CW
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif orientation == 7:  # Transverse
            image = cv2.transpose(image)
            image = cv2.flip(image, 1)
        elif orientation == 8:  # 90 CCW
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        return image
    
    def convert_color_profile(self, image: np.ndarray, metadata: Dict) -> np.ndarray:
        """
        Convert from iPhone's P3 color space to sRGB if needed
        
        Args:
            image: Image array
            metadata: Metadata dictionary
            
        Returns:
            Color-corrected image
        """
        # iPhone typically uses Display P3 color space
        # Simple approximation of P3 to sRGB conversion
        model = metadata.get('model', '')
        
        if 'iPhone' in str(model):
            # Apply simple P3 to sRGB transformation
            # This is an approximation; proper color management would use ICC profiles
            image_float = image.astype(np.float32) / 255.0
            
            # P3 to sRGB matrix (simplified)
            # More accurate would use proper color management
            transform = np.array([
                [1.2249, -0.2247, 0.0],
                [-0.0420, 1.0419, 0.0],
                [-0.0197, -0.0786, 1.0983]
            ])
            
            # Apply transformation
            shape = image_float.shape
            pixels = image_float.reshape(-1, 3)
            pixels_srgb = pixels @ transform.T
            image_srgb = pixels_srgb.reshape(shape)
            
            # Clip and convert back
            image = np.clip(image_srgb * 255, 0, 255).astype(np.uint8)
        
        return image
    
    def get_adaptive_parameters(self, metadata: Dict) -> Dict:
        """
        Determine adaptive processing parameters based on metadata
        
        Args:
            metadata: Image metadata
            
        Returns:
            Dictionary of processing parameters
        """
        params = {
            'gamma': 1.0,
            'clahe_clip': 3.0,
            'denoise_strength': 0,
            'sharpen_strength': 0,
            'white_balance_correction': False
        }
        
        hints = metadata.get('lighting_hints', {})
        
        # Adjust for harsh sunlight
        if hints.get('likely_harsh_sun'):
            params['gamma'] = 0.7
            params['clahe_clip'] = 2.0
            logger.debug("Detected harsh sunlight from metadata")
        
        # Adjust for low light
        elif hints.get('likely_low_light'):
            params['gamma'] = 1.3
            params['clahe_clip'] = 4.0
            params['denoise_strength'] = 10  # Higher ISO = more noise
            logger.debug("Detected low light from metadata")
        
        # Flash compensation
        if hints.get('flash_used'):
            params['gamma'] = 0.9
            params['white_balance_correction'] = True
            logger.debug("Flash detected, adjusting parameters")
        
        # Exposure compensation
        exp_comp = hints.get('exposure_compensation', 0)
        if exp_comp < -0.5:
            params['gamma'] *= 1.2  # Image was underexposed
        elif exp_comp > 0.5:
            params['gamma'] *= 0.8  # Image was overexposed
        
        return params
    
    def preprocess_iphone_image(self, image: np.ndarray, 
                               metadata: Dict) -> np.ndarray:
        """
        Apply iPhone-specific preprocessing based on metadata
        
        Args:
            image: Input image
            metadata: Image metadata
            
        Returns:
            Preprocessed image
        """
        params = self.get_adaptive_parameters(metadata)
        result = image.copy()
        
        # Denoise if needed (high ISO)
        if params['denoise_strength'] > 0:
            result = cv2.fastNlMeansDenoisingColored(
                result, None, 
                params['denoise_strength'], 
                params['denoise_strength'], 
                7, 21
            )
        
        # White balance correction if flash was used
        if params['white_balance_correction']:
            result = self.correct_flash_white_balance(result)
        
        # Apply gamma correction if needed
        if params['gamma'] != 1.0:
            result = self.apply_gamma(result, params['gamma'])
        
        return result
    
    def correct_flash_white_balance(self, image: np.ndarray) -> np.ndarray:
        """
        Correct white balance issues from flash photography
        
        Args:
            image: Input image
            
        Returns:
            White balance corrected image
        """
        # Simple gray world white balance
        result = image.astype(np.float32)
        avg = result.mean(axis=(0, 1))
        gray = avg.mean()
        
        # Calculate scaling factors
        scale = gray / (avg + 1e-6)
        scale = np.clip(scale, 0.8, 1.2)  # Limit correction
        
        # Apply scaling
        for c in range(3):
            result[:, :, c] *= scale[c]
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def apply_gamma(self, image: np.ndarray, gamma: float) -> np.ndarray:
        """
        Apply gamma correction
        
        Args:
            image: Input image
            gamma: Gamma value
            
        Returns:
            Gamma corrected image
        """
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in range(256)]).astype(np.uint8)
        return cv2.LUT(image, table)
    
    def fallback_load(self, file_path: str) -> Tuple[np.ndarray, Dict]:
        """
        Fallback method if HEIC support is not available
        
        Args:
            file_path: Path to image file
            
        Returns:
            Tuple of (image_array, empty_metadata)
        """
        # Try loading with OpenCV (won't work for HEIC but handles other formats)
        try:
            img = cv2.imread(file_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return img, {'format': 'unknown', 'has_metadata': False}
        except:
            pass
        
        # Return empty image and metadata
        logger.error(f"Could not load image: {file_path}")
        return np.zeros((384, 384, 3), dtype=np.uint8), {'format': 'error', 'has_metadata': False}
    
    def process(self, file_path: str) -> Tuple[np.ndarray, Dict]:
        """
        Main processing method for iPhone images
        
        Args:
            file_path: Path to image file
            
        Returns:
            Tuple of (processed_image, metadata)
        """
        path = Path(file_path)
        
        # Check if HEIC format
        if path.suffix.lower() in self.supported_extensions:
            image, metadata = self.load_heic(file_path)
        else:
            # Load as regular image
            image = cv2.imread(file_path)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            metadata = {'format': path.suffix, 'has_metadata': False}
        
        # Apply iPhone-specific preprocessing if metadata available
        if metadata.get('has_metadata') and image is not None:
            image = self.preprocess_iphone_image(image, metadata)
        
        return image, metadata