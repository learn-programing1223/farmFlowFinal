# Test-Time Augmentation

## Requirements
- 12-15 augmentations with 1-2s budget
- 2-3% accuracy improvement
- 600-900ms processing time

## Augmentation Types
- Geometric: rotations, flips
- Photometric: brightness, contrast
- Scale: 0.8× to 1.2×

## Aggregation
- Confidence-weighted averaging
- 1.5% better than simple averaging