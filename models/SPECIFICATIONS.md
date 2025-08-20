# Model Specifications

## Three-Tier Cascade Architecture

### Overview
Progressive inference system balancing speed and accuracy based on confidence levels.

### Tier 1: EfficientFormer-L7
**Purpose**: Rapid initial classification for clear cases

| Metric | Target | Actual | Notes |
|--------|--------|--------|-------|
| Inference Time | <10ms | 7ms | On iPhone 12 Neural Engine |
| Accuracy | 95%+ | TBD | Simple disease cases |
| Parameters | ~10M | 10.2M | Optimized for mobile |
| Confidence Threshold | 85% | Tunable | For escalation decision |

**When to Use**:
- Clear, well-lit images
- Single disease presentation
- High-confidence cases
- Battery-conscious mode

### Tier 2: EfficientNet-B4
**Purpose**: High-accuracy classification for complex cases

| Metric | Target | Actual | Notes |
|--------|--------|--------|-------|
| Inference Time | 600-800ms | 700ms | Full preprocessing |
| Accuracy | 99.91% | TBD | Laboratory conditions |
| Parameters | 19M | 19.3M | Balance of size/accuracy |
| Field Accuracy | 85%+ | TBD | Real-world conditions |

**When to Use**:
- Moderate confidence from Tier 1
- Complex disease presentations
- Standard operation mode
- When accuracy is critical

### Tier 3: Hybrid Ensemble
**Purpose**: Maximum accuracy for ambiguous cases

| Metric | Target | Actual | Notes |
|--------|--------|--------|-------|
| Inference Time | 1.2-1.5s | TBD | Including TTA |
| Accuracy | 99.9%+ | TBD | With uncertainty |
| Components | 3-5 models | TBD | To be determined |
| Memory | <300MB | TBD | Peak usage |

**Ensemble Strategy Options**:
1. **Voting Ensemble**: Multiple models vote
2. **Stacking**: Meta-learner combines predictions
3. **Boosting**: Sequential refinement
4. **Mixture of Experts**: Specialized models

**When to Use**:
- Low confidence from Tier 2
- Unknown disease detection
- Research/validation mode
- Critical diagnoses

## Cascade Decision Logic

```
Input Image → Preprocessing → Tier 1
    ↓
If confidence > 85%: Return prediction
    ↓
Else → Tier 2
    ↓
If confidence > 70%: Return prediction
    ↓
Else → Tier 3 + Uncertainty
    ↓
If confidence > 50%: Return with warning
    ↓
Else: Flag as Unknown
```

## Model Training Strategy

### Pretraining
- **Dataset**: PlantCLEF2022 (1M+ images)
- **Approach**: Self-supervised learning
- **Augmentation**: CycleGAN for domain adaptation

### Fine-tuning
- **Laboratory images**: 25,000
- **Field images**: 5,000 genuine iPhone photos
- **Unknown samples**: 2,000
- **Validation**: Group-aware splitting

### Data Augmentation (Critical)
- **CycleGAN**: Prevents 45-68% field accuracy drop
- **Runtime augmentation**: TTA for uncertainty
- **Synthetic diseases**: Generate edge cases

## Optimization Requirements

### Core ML Conversion
- Float16 quantization (NOT int8)
- Pruning: 25% sparsity target
- Palettization: 6-bit weights
- Target size: <52MB per model

### Memory Management
- Peak usage: <300MB
- Model swapping: Tier-based loading
- Cache management: LRU for features

### Thermal Management
- Monitor device temperature
- Downgrade to Tier 1 if >45°C
- Batch processing delays
- 3.2% battery/hour target

## Disease Categories

### Primary Classes (6)
1. **Healthy**: No disease present
2. **Blight**: Early/Late blight
3. **Leaf Spot**: Bacterial/Fungal spots
4. **Powdery Mildew**: White fungal growth
5. **Mosaic Virus**: Viral patterns

### Special Class (NOT trained)
**Unknown**: Novel/ambiguous diseases (detected via uncertainty)

## Performance Benchmarks

### Laboratory Conditions
- Accuracy: 99.2%
- Precision: 99.1%
- Recall: 99.3%
- F1 Score: 99.2%

### Field Conditions by Lighting
| Condition | Target | Strategy |
|-----------|--------|----------|
| Direct Sunlight | 79% | Heavy preprocessing |
| Cloudy/Diffuse | 86% | Standard pipeline |
| Shade/Partial | 81% | Illumination correction |
| Indoor/Greenhouse | 89% | Minimal preprocessing |
| Low Light | 73% | Super-resolution focus |

## Validation Requirements

### Cross-validation
- 5-fold stratified CV
- Group-aware (no plant leakage)
- Time-based (historical validation)

### Field Testing
- 100+ real farm deployments
- Diverse geographic regions
- Multiple crop types
- Seasonal variations

### Edge Case Testing
- Blurry images
- Multiple diseases
- Partial occlusion
- Wet conditions

## Implementation Freedom
Let Claude Code determine:
- Optimal ensemble strategy
- Best pretrained weights
- Augmentation parameters
- Confidence thresholds
- Architecture modifications

Within these performance constraints