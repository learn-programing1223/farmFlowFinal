# EfficientFormer-L7 Requirements

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

## TIER 1 MODEL SPECIFICATIONS
- **Role**: Fast initial classification
- **Inference time**: 7ms on iPhone 12
- **Accuracy target**: 95%+
- **Use case**: Simple, clear disease cases
- **Parameters**: ~10M

## RESEARCH FINDINGS
- Achieves 83.3% ImageNet accuracy
- Leaves budget for preprocessing/ensemble
- Efficient attention mechanisms optimized for Neural Engine
- Superior to MobileNetV3 for this use case
- Better accuracy/speed ratio than EfficientNet-Lite

## CASCADE INTEGRATION
- Must provide confidence scores for escalation decision
- Confidence threshold: 85% for immediate classification
- Below threshold: Escalate to Tier 2
- Must support Monte Carlo Dropout for uncertainty

## IMPLEMENTATION REQUIREMENTS
- Use pretrained weights from timm library
- Custom disease classification head (8 classes)
- Dropout layers for uncertainty quantification
- Output both class predictions and confidence scores

## OPTIMIZATION TARGETS
- Core ML compatible architecture
- Float16 quantization ready
- Batch size 1 optimized (real-time inference)
- Memory footprint <50MB

## CONSTRAINTS
- Must integrate with cascade system
- Must provide confidence scores for escalation
- Must support uncertainty quantification
- Cannot exceed 10ms inference on A14 chip

## Validation Metrics
- Measure inference time on actual iPhone hardware
- Test accuracy on simple vs complex disease cases
- Validate confidence calibration
- Check memory usage during inference

## Related Files
- Research document: Model Architecture section
- Cascade logic: /inference/pipeline.md
- Core ML conversion: /ios/CoreMLConversion/

## Implementation Freedom
Let Claude Code determine optimal architecture details within these constraints