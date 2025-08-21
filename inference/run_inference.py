"""
Complete End-to-End Inference Pipeline
Processes plant images through preprocessing and model cascade
Returns disease predictions with confidence scores
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import time
import json
from typing import Dict, Optional, Tuple, List
import argparse
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import preprocessing components
from preprocessing.lassr import LASSRProcessor
from preprocessing.illumination.retinex_illumination import RetinexIllumination
from preprocessing.segmentation.cascade_controller import SegmentationCascade
from preprocessing.pipeline_integration import PreprocessingPipeline

# Import model components
from models.cascade.cascade_controller import ModelCascadeController

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PlantDiseaseInference:
    """
    Complete inference pipeline for plant disease detection
    Integrates preprocessing and model cascade
    """
    
    def __init__(self, 
                 enable_preprocessing: bool = True,
                 enable_tier2: bool = True,
                 enable_tier3: bool = False,
                 device: str = 'cpu',
                 verbose: bool = True):
        """
        Initialize inference pipeline
        
        Args:
            enable_preprocessing: Run preprocessing pipeline
            enable_tier2: Enable Tier 2 model
            enable_tier3: Enable Tier 3 model
            device: Computation device
            verbose: Print detailed logs
        """
        self.enable_preprocessing = enable_preprocessing
        self.verbose = verbose
        self.device = device
        
        # Initialize preprocessing if enabled
        if enable_preprocessing:
            logger.info("Initializing preprocessing pipeline...")
            self.preprocessing = PreprocessingPipeline(
                enable_lassr=True,
                enable_illumination=True,
                enable_segmentation=True
            )
        else:
            self.preprocessing = None
            
        # Initialize model cascade
        logger.info("Initializing model cascade...")
        self.model_cascade = ModelCascadeController(
            enable_tier2=enable_tier2,
            enable_tier3=enable_tier3,
            device=device
        )
        
        # Disease class names
        self.disease_classes = [
            'Healthy',
            'Blight',
            'Leaf Spot',
            'Powdery Mildew',
            'Mosaic Virus',
            'Nutrient Deficiency',
            'Unknown'
        ]
        
        # Timing statistics
        self.timing_stats = {
            'preprocessing': [],
            'inference': [],
            'total': []
        }
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load and validate image
        
        Args:
            image_path: Path to image file
            
        Returns:
            RGB image array
        """
        # Check if file exists
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Log image info
        if self.verbose:
            h, w = image.shape[:2]
            logger.info(f"Loaded image: {w}x{h} pixels")
        
        return image
    
    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Run preprocessing pipeline
        
        Args:
            image: Input RGB image
            
        Returns:
            Tuple of (processed_image, preprocessing_info)
        """
        if self.preprocessing is None:
            # No preprocessing, just resize
            target_size = 384
            if image.shape[:2] != (target_size, target_size):
                image = cv2.resize(image, (target_size, target_size))
            return image, {'preprocessing_applied': False}
        
        # Run full preprocessing pipeline
        start_time = time.time()
        
        result = self.preprocessing.process(image)
        
        preprocessing_time = (time.time() - start_time) * 1000
        
        # Extract processed image
        if 'segmented' in result:
            processed_image = result['segmented']
        elif 'illumination_corrected' in result:
            processed_image = result['illumination_corrected']
        elif 'super_resolved' in result:
            processed_image = result['super_resolved']
        else:
            processed_image = image
        
        # Build info dictionary
        info = {
            'preprocessing_applied': True,
            'time_ms': preprocessing_time,
            'disease_preserved': result.get('disease_preservation', 1.0),
            'quality_score': result.get('quality_score', 0.0),
            'steps_applied': []
        }
        
        # Track which steps were applied
        if 'super_resolved' in result:
            info['steps_applied'].append('LASSR')
        if 'illumination_corrected' in result:
            info['steps_applied'].append('Retinex')
        if 'segmented' in result:
            info['steps_applied'].append('Segmentation')
        
        self.timing_stats['preprocessing'].append(preprocessing_time)
        
        if self.verbose:
            logger.info(f"Preprocessing complete: {preprocessing_time:.1f}ms")
            logger.info(f"Steps applied: {', '.join(info['steps_applied'])}")
        
        return processed_image, info
    
    def run_inference(self, image: np.ndarray, 
                     preprocessed_data: Optional[Dict] = None) -> Dict:
        """
        Run model cascade inference
        
        Args:
            image: Preprocessed image
            preprocessed_data: Optional preprocessing metadata
            
        Returns:
            Inference results
        """
        start_time = time.time()
        
        # Run cascade
        result = self.model_cascade.infer(image, preprocessed_data)
        
        inference_time = (time.time() - start_time) * 1000
        self.timing_stats['inference'].append(inference_time)
        
        # Add disease name
        if result['class'] not in self.disease_classes:
            # Map to Unknown if not recognized
            if result['confidence'] < 0.70:
                result['class'] = 'Unknown'
        
        # Add inference time
        result['inference_time_ms'] = inference_time
        
        if self.verbose:
            logger.info(f"Inference complete: {inference_time:.1f}ms")
            logger.info(f"Prediction: {result['class']} (confidence: {result['confidence']:.2f})")
            logger.info(f"Cascade path: {result.get('cascade_path', [])}")
        
        return result
    
    def process_image(self, image_path: str) -> Dict:
        """
        Complete pipeline: load, preprocess, and run inference
        
        Args:
            image_path: Path to plant image
            
        Returns:
            Complete results dictionary
        """
        total_start = time.time()
        
        # Load image
        try:
            image = self.load_image(image_path)
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            return {
                'success': False,
                'error': str(e),
                'image_path': image_path
            }
        
        # Preprocess
        if self.enable_preprocessing:
            processed_image, preprocessing_info = self.preprocess_image(image)
        else:
            processed_image = image
            preprocessing_info = {'preprocessing_applied': False}
        
        # Run inference
        inference_result = self.run_inference(processed_image, preprocessing_info)
        
        # Calculate total time
        total_time = (time.time() - total_start) * 1000
        self.timing_stats['total'].append(total_time)
        
        # Build complete result
        result = {
            'success': True,
            'image_path': image_path,
            'prediction': inference_result['class'],
            'confidence': inference_result['confidence'],
            'all_probabilities': inference_result.get('all_probabilities', []),
            'cascade_path': inference_result.get('cascade_path', []),
            'timing': {
                'preprocessing_ms': preprocessing_info.get('time_ms', 0),
                'inference_ms': inference_result['inference_time_ms'],
                'total_ms': total_time
            },
            'preprocessing': preprocessing_info,
            'tier_used': inference_result.get('tier', 0),
            'should_escalate': inference_result.get('should_escalate', False)
        }
        
        # Safety check - enforce Unknown for low confidence
        if result['confidence'] < 0.70:
            result['prediction'] = 'Unknown'
            result['safety_override'] = True
            if self.verbose:
                logger.info("Safety: Low confidence detected, classifying as Unknown")
        
        return result
    
    def process_batch(self, image_paths: List[str]) -> List[Dict]:
        """
        Process multiple images
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of results
        """
        results = []
        
        for i, path in enumerate(image_paths):
            if self.verbose:
                logger.info(f"\nProcessing image {i+1}/{len(image_paths)}: {Path(path).name}")
            
            result = self.process_image(path)
            results.append(result)
        
        # Print summary statistics
        if self.verbose and len(results) > 0:
            self.print_batch_summary(results)
        
        return results
    
    def print_batch_summary(self, results: List[Dict]):
        """Print summary statistics for batch processing"""
        successful = [r for r in results if r.get('success', False)]
        
        if len(successful) == 0:
            logger.warning("No successful predictions")
            return
        
        # Prediction distribution
        predictions = {}
        for r in successful:
            pred = r['prediction']
            predictions[pred] = predictions.get(pred, 0) + 1
        
        # Timing statistics
        avg_preprocess = np.mean([r['timing']['preprocessing_ms'] for r in successful])
        avg_inference = np.mean([r['timing']['inference_ms'] for r in successful])
        avg_total = np.mean([r['timing']['total_ms'] for r in successful])
        
        # Confidence statistics
        confidences = [r['confidence'] for r in successful]
        avg_confidence = np.mean(confidences)
        
        # Cascade statistics
        cascade_stats = {'tier1': 0, 'tier2': 0, 'tier3': 0}
        for r in successful:
            tier = r.get('tier_used', 1)
            cascade_stats[f'tier{tier}'] = cascade_stats.get(f'tier{tier}', 0) + 1
        
        print("\n" + "="*60)
        print("BATCH PROCESSING SUMMARY")
        print("="*60)
        
        print(f"\nProcessed: {len(successful)}/{len(results)} images successfully")
        
        print("\nPrediction Distribution:")
        for disease, count in sorted(predictions.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(successful)) * 100
            print(f"  {disease}: {count} ({percentage:.1f}%)")
        
        print(f"\nTiming (average):")
        print(f"  Preprocessing: {avg_preprocess:.1f}ms")
        print(f"  Inference: {avg_inference:.1f}ms")
        print(f"  Total: {avg_total:.1f}ms")
        
        print(f"\nConfidence:")
        print(f"  Average: {avg_confidence:.3f}")
        print(f"  Min: {min(confidences):.3f}")
        print(f"  Max: {max(confidences):.3f}")
        
        print(f"\nCascade Usage:")
        for tier, count in cascade_stats.items():
            if count > 0:
                print(f"  {tier}: {count} images")
        
        print("="*60)
    
    def save_results(self, results: Dict, output_path: str):
        """Save results to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        if self.verbose:
            logger.info(f"Results saved to: {output_path}")


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Plant Disease Detection Inference')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--batch', type=str, help='Path to directory of images')
    parser.add_argument('--output', type=str, help='Output JSON file for results')
    parser.add_argument('--no-preprocessing', action='store_true', 
                       help='Skip preprocessing pipeline')
    parser.add_argument('--no-tier2', action='store_true', 
                       help='Disable Tier 2 model')
    parser.add_argument('--enable-tier3', action='store_true',
                       help='Enable Tier 3 model')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device for inference (cpu/cuda)')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = PlantDiseaseInference(
        enable_preprocessing=not args.no_preprocessing,
        enable_tier2=not args.no_tier2,
        enable_tier3=args.enable_tier3,
        device=args.device,
        verbose=not args.quiet
    )
    
    # Process image(s)
    if args.image:
        # Single image
        result = pipeline.process_image(args.image)
        
        # Save if output specified
        if args.output:
            pipeline.save_results(result, args.output)
        else:
            # Print result
            print(json.dumps(result, indent=2))
    
    elif args.batch:
        # Batch processing
        batch_dir = Path(args.batch)
        
        # Find all images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(batch_dir.glob(f'*{ext}'))
            image_paths.extend(batch_dir.glob(f'*{ext.upper()}'))
        
        if len(image_paths) == 0:
            print(f"No images found in: {batch_dir}")
            return
        
        print(f"Found {len(image_paths)} images")
        
        # Process batch
        results = pipeline.process_batch([str(p) for p in image_paths])
        
        # Save if output specified
        if args.output:
            pipeline.save_results(results, args.output)
    
    else:
        print("Please specify --image or --batch")
        parser.print_help()


if __name__ == "__main__":
    main()