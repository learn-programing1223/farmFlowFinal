# Evaluation Metrics

## Cross-Species Validation

### Critical Tests
- Test same disease on plants NOT in training set
- Verify model detects disease patterns, not memorizing plants
- Example: Train on tomato blight, test on pepper blight

### Performance Metrics
- Accuracy per disease (across all species)
- NOT accuracy per plant type
- Cross-species generalization score

## Target Performance
- Laboratory accuracy: 99.2%
- Field accuracy: 82-87%
- Unknown detection: 94% precision, 91% recall
- Inference time: 950ms average

## Field Performance by Condition
- Direct sunlight: 79%
- Cloudy: 86%
- Shade: 81%
- Indoor: 89%
- Low light: 73%

## Unknown Detection
- Tested on OOD samples (diseases not in 7 classes)
- NOT a trained class - confidence-based