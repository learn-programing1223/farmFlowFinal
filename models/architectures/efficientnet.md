# EfficientNet-B4 (Tier 2 Model)

## Disease Pattern Focus

Model should learn to detect:
- Disease visual symptoms (NOT plant features)
- Cross-species disease patterns
- Texture abnormalities
- Color deviations from healthy
- Lesion patterns and progression

The same disease classifier works on ANY plant:
- Powdery mildew (roses, cucumbers, grapes, squash, etc.)
- Blight (tomatoes, potatoes, peppers, eggplants, etc.)
- Mosaic virus (ANY affected plant)
- ALL diseases detected universally across species

## Role in Cascade
- Second tier: High-confidence classification
- Activated when Tier 1 confidence < 85%
- Primary workhorse for most disease detection

## Performance Requirements (from research)
- Inference time: 600-800ms on iPhone 12
- Accuracy: 99.91% on PlantVillage
- Parameters: 19M
- Model size: ~15-20MB after optimization

## Architecture Specifications
- EfficientNet-B4 variant
- Compound scaling methodology
- Input resolution: 384×384 optimal
- Can handle 512×512 for maximum accuracy

## Key Features
- Better for complex disease patterns than Tier 1
- Provides confidence scores for uncertainty
- Must support Monte Carlo Dropout
- Pretrained on PlantCLEF2022 preferred

## Integration Requirements
- Works within cascade system
- Provides uncertainty metrics
- Compatible with Core ML conversion
- Supports Float16 quantization

## When This Model is Used
- Tier 1 confidence between 70-85%
- Complex disease presentations
- Multiple potential diseases detected
- Need for higher accuracy than speed

Related:
- Research document: Model Architecture section
- Cascade: /inference/pipeline.md
- Tier 1: /models/architectures/efficientformer.md