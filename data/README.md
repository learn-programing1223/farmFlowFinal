# Data Directory

## Structure
- `raw/`: Original images (HEIC, JPEG)
- `processed/`: Preprocessed images ready for training

## Dataset Diversity Requirements

### Critical: Multi-Species for Each Disease
Each disease class MUST include multiple plant species:

Example for "Powdery Mildew" class:
- Powdery mildew on roses
- Powdery mildew on cucumbers  
- Powdery mildew on grapes
- Powdery mildew on squash

This ensures model learns DISEASE PATTERNS not plant-specific features.

### Dataset Composition
- 6 classes total (5 diseases + Healthy)
- Each disease on 5+ different plant species
- Various stages of disease progression
- Different lighting/backgrounds

### What We're Training
✓ Visual disease symptoms
✓ Pattern recognition across species
✓ Disease progression stages
✗ NOT plant identification
✗ NOT species-specific models

## Dataset Requirements (from research)
- 25,000 laboratory/controlled images
- 5,000 genuine iPhone field photos
- 1,000 images per disease category
- 2,000 out-of-distribution samples for validating unknown detection thresholds(These are NOT used for training - only for testing uncertainty calibration)
- 1,000 edge cases (blur, occlusion, extreme conditions)

## Data Pipeline
1. Load raw images → `/loaders.py`
2. Preprocess → `/preprocessing/pipeline.py`
3. Split (group-aware) → `/splitters.py`
4. Augment (runtime only) → `/augmentation.py`

⚠️ CRITICAL: Never let images from same plant/location appear in train and test sets!