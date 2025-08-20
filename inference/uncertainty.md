# Uncertainty Quantification Requirements

## THREE-TIER APPROACH (from research)

### 1. Epistemic Uncertainty (Model Uncertainty)
- **Method**: Monte Carlo Dropout
- **Iterations**: 30 forward passes
- **Time budget**: 100-150ms
- **Purpose**: Capture model's knowledge uncertainty

### 2. Aleatoric Uncertainty (Data Uncertainty)
- **Method**: Learned variance prediction
- **Architecture**: Dual-head output (mean + variance)
- **Time budget**: Negligible (single pass)
- **Purpose**: Capture inherent data noise

### 3. Distribution Uncertainty (Out-of-Distribution)
- **Method**: One-class SVM on features
- **Training**: On known disease features only
- **Time budget**: 50-75ms
- **Purpose**: Detect unknown diseases

## DECISION THRESHOLDS

### Confidence-Based Actions
| Confidence | Action | Rationale |
|------------|--------|-----------|
| >85% | Immediate classification | High certainty, no refinement needed |
| 70-85% | Refine with augmentation | Moderate certainty, can improve |
| <70% | Flag as Unknown | Too uncertain, need expert review |

### Uncertainty Components
```
Total Uncertainty = α·Epistemic + β·Aleatoric + γ·Distribution
where α=0.4, β=0.3, γ=0.3 (tunable)
```

## PERFORMANCE TARGETS

### Unknown Detection
- **Precision**: 94% (minimize false unknowns)
- **Recall**: 91% (catch most unknowns)
- **F1 Score**: 92.5%

### Known Disease Classification
- **When confident (>85%)**: 99%+ accuracy
- **After refinement (70-85%)**: 96%+ accuracy
- **Overall accuracy**: 95%+

## IMPLEMENTATION REQUIREMENTS

### Monte Carlo Dropout
- Enable dropout during inference
- Use consistent dropout rate (0.2)
- Aggregate predictions with variance calculation
- Cache results for efficiency

### Learned Variance
- Modify final layer for dual output
- Train with negative log-likelihood loss
- Validate calibration on hold-out set

### One-class SVM
- Extract features from penultimate layer
- Train on "normal" disease patterns
- Set decision boundary at 95th percentile
- Update periodically with new diseases

## VALIDATION METRICS

### Calibration
- Expected Calibration Error (ECE) < 0.05
- Maximum Calibration Error (MCE) < 0.10
- Reliability diagrams for each disease

### Unknown Detection
- Confusion matrix for known vs unknown
- ROC curve for threshold selection
- Disease-specific performance

### Runtime Performance
- Total uncertainty calculation < 250ms
- Memory usage < 50MB additional
- Battery impact < 0.5% per inference

## EDGE CASES

### Handle These Scenarios
- Blurry images → High aleatoric uncertainty
- New disease variants → High distribution uncertainty
- Ambiguous symptoms → High epistemic uncertainty
- Multiple diseases → Multi-label uncertainty

## INTEGRATION POINTS
- Cascade inference system (/inference/pipeline.md)
- Model architectures (/models/architectures/)
- Evaluation metrics (/evaluation/metrics.md)

## Related Research
- Uncertainty Quantification section in research doc
- Gal & Ghahramani (2016) on MC Dropout
- Kendall & Gal (2017) on uncertainty types

## Implementation Freedom
Let Claude Code design optimal uncertainty system within these constraints