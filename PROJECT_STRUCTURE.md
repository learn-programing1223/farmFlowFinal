# FarmFlow Project Navigation

## Quick Links
- **Training Entry Point**: `/train.py`
- **Preprocessing Pipeline**: `/preprocessing/pipeline.py` (25-35% accuracy)
- **Model Architectures**: `/models/architectures/`
- **iOS Conversion**: `/ios/CoreMLConversion/convert.py`
- **Evaluation**: `/evaluation/metrics.py`

## Critical Files
- **LASSR Implementation**: `/preprocessing/lassr.py` (21% accuracy gain)
- **CycleGAN**: `/preprocessing/cyclegan.py` (prevents 45-68% field drop)
- **U-Net Segmentation**: `/preprocessing/segmentation/unet.py` (30-40% gain)
- **Uncertainty**: `/inference/uncertainty.md` (Detects Unknown via confidence thresholds - NOT a trained class)

## Performance Targets
- Lab: 99.2% accuracy
- Field: 82-87% accuracy  
- Unknown: 94% precision, 91% recall
- Inference: 950ms average

## Current Status
See `/STATUS.md` for development progress