# Unknown Detection Strategy

## CRITICAL: Unknown is NOT a trained class

### How it Works
1. Model outputs 7 class probabilities (softmax)
2. Calculate uncertainty metrics
3. If any threshold exceeded → return "Unknown"

### Thresholds (calibrated on validation)
- Confidence < 70%
- Entropy > 0.5 (normalized)
- MC Dropout variance > 0.2
- OOD score > threshold

### Validation Data
The 2,000 "unknown" samples are:
- Diseases NOT in our 6 disease categories
- Used to calibrate thresholds
- NOT used for training
- Examples: Rust, Bacterial Spot, Canker, etc.

### Why This is Better
- Handles ANY unexpected input
- No need to define what "unknown" looks like
- More theoretically sound
- Can adjust thresholds without retraining

## Detailed Detection Methodology

### 1. Softmax Confidence
```
max_confidence = max(softmax_outputs)
if max_confidence < 0.70:
    return "Unknown"
```
- Simple but effective first filter
- Catches obvious uncertainty
- Fast computation (no extra forward passes)

### 2. Entropy-Based Detection
```
entropy = -sum(p * log(p) for p in softmax_outputs)
normalized_entropy = entropy / log(num_classes)
if normalized_entropy > 0.5:
    return "Unknown"
```
- Measures prediction uncertainty
- High entropy = model is confused
- Normalized for consistency

### 3. Monte Carlo Dropout Variance
```
predictions = [model(x, training=True) for _ in range(30)]
variance = var(predictions, axis=0)
if max(variance) > 0.2:
    return "Unknown"
```
- 30 forward passes with dropout
- High variance = epistemic uncertainty
- Catches model knowledge gaps

### 4. Out-of-Distribution Detection
```
features = model.get_features(x)  # Penultimate layer
ood_score = one_class_svm.decision_function(features)
if ood_score < calibrated_threshold:
    return "Unknown"
```
- Trained on known disease features only
- Detects inputs far from training distribution
- Complementary to other methods

## Calibration Process

### Step 1: Collect Metrics
For each validation image (including unknowns):
1. Record max confidence
2. Calculate entropy
3. Compute MC dropout variance
4. Get OOD score

### Step 2: Optimize Thresholds
```python
# Optimize for target metrics:
# - Unknown Precision: 94%
# - Unknown Recall: 91%
# - Known Accuracy: >95%

for threshold in candidate_thresholds:
    precision, recall = evaluate(threshold)
    if precision >= 0.94 and recall >= 0.91:
        selected_threshold = threshold
```

### Step 3: Validate Performance
- Test on held-out unknown diseases
- Check false positive rate on known diseases
- Verify across lighting conditions
- Ensure consistency across disease stages

## Unknown Sample Categories

### Training Data (0 unknowns)
- 6 known disease categories
- Healthy plants
- Total: 25,000 lab + 5,000 field images

### Validation Unknowns (2,000 samples)
Used ONLY for threshold calibration:
- **Rust diseases** (500 samples)
- **Bacterial spots** (400 samples)
- **Viral infections** (not Mosaic) (300 samples)
- **Canker diseases** (300 samples)
- **Root diseases** (visible on leaves) (200 samples)
- **Pest damage** (mistaken for disease) (200 samples)
- **Physical damage** (frost, hail) (100 samples)

### Test Unknowns (500 samples)
Completely held out for final evaluation:
- Novel diseases not in validation
- Extreme presentations of known diseases
- Multiple concurrent diseases
- Non-plant objects

## Advantages Over Training "Unknown" Class

### 1. Theoretical Soundness
- Unknown is absence of knowledge, not a class
- Can't train on "everything else"
- Uncertainty is the right framework

### 2. Flexibility
- Adjust thresholds without retraining
- Add new uncertainty metrics easily
- Adapt to deployment feedback

### 3. Robustness
- Handles completely novel inputs
- Not limited to seen "unknowns"
- Graceful degradation

### 4. Interpretability
- Know WHY something is unknown
- Can provide uncertainty breakdown
- Helps users understand limitations

## Implementation Priority

### Phase 1: Basic Confidence
- Implement softmax threshold only
- Quick deployment, immediate value
- Establish baseline performance

### Phase 2: Add Entropy
- Combine confidence + entropy
- Better theoretical foundation
- Improved detection

### Phase 3: MC Dropout
- Add epistemic uncertainty
- More compute but better results
- Critical for ambiguous cases

### Phase 4: OOD Detection
- Complete uncertainty picture
- Catches distribution shift
- Production-ready system

## Performance Targets

### Detection Metrics
- **Unknown Precision**: 94% (few false unknowns)
- **Unknown Recall**: 91% (catch most unknowns)
- **F1 Score**: 92.5%

### Operational Metrics
- **Detection time**: <250ms additional
- **Memory overhead**: <50MB
- **Battery impact**: Negligible

### Confidence Calibration
- Expected Calibration Error < 0.05
- Maximum Calibration Error < 0.10
- Reliability across all conditions

## Edge Cases and Handling

### Multiple Diseases
- Often triggers unknown (correct behavior)
- High entropy from split predictions
- User directed to expert

### Blurry Images
- High aleatoric uncertainty
- May trigger unknown (desired)
- Prompt for better image

### Novel Presentations
- Known disease, unusual appearance
- OOD detection should catch
- Valuable for model improvement

### Adversarial Inputs
- Non-plant images
- Extreme OOD scores
- Clear unknown detection

## Monitoring and Improvement

### Collect Metrics
- Log all uncertainty scores
- Track unknown detections
- Monitor threshold performance

### Periodic Recalibration
- Monthly threshold adjustment
- Based on deployment data
- Balance precision/recall

### Model Updates
- Retrain with new diseases
- Add to known categories
- Reduce unknown rate over time

## User Communication

### When Returning Unknown
```
"Unknown Disease Detected
Confidence: 45%
Recommendation: Consult expert
Reason: Unusual symptoms not in database"
```

### Provide Context
- Show confidence breakdown
- Explain why unknown
- Suggest next steps
- Option to submit for review

## Integration with Cascade

### Tier 1 (EfficientFormer)
- Basic confidence check only
- Fast unknown detection
- Escalate uncertain cases

### Tier 2 (EfficientNet)
- Add entropy calculation
- More sophisticated detection
- Balance speed/accuracy

### Tier 3 (Ensemble)
- Full uncertainty suite
- MC Dropout + OOD
- Maximum detection capability

This strategy ensures robust unknown detection without the theoretical problems of training an "unknown" class.