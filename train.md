# Training Pipeline

## Training Strategy for Disease Patterns

### Key Principle
Train on disease symptoms, NOT plant features

### Data Organization
```
/data/
  /Blight/
    - tomato_blight_001.jpg
    - potato_blight_001.jpg
    - pepper_blight_001.jpg
  /Powdery_Mildew/
    - rose_pm_001.jpg
    - cucumber_pm_001.jpg
    - grape_pm_001.jpg
```

### Loss Function Considerations
- Standard CrossEntropy on 6 classes (5 diseases + Healthy)
- No species-specific weighting
- Focus on disease feature extraction

## Key Requirements
- Train on 6 classes only (Unknown is confidence-based)
- Use group-aware splitting (prevent leakage)
- Apply CycleGAN augmentation
- Calibrate uncertainty thresholds post-training

## Dataset
- 25,000 lab images
- 5,000 field images
- 2,000 OOD samples for threshold calibration

## Critical
- Never train Unknown as a class
- Fit preprocessing only on training data