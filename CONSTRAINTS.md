# Project Constraints

## Hard Requirements (Non-negotiable)

### Performance Constraints
- **Total inference time**: <2 seconds (including all preprocessing)
- **Average inference**: 950ms target
- **Peak inference**: 1.8 seconds maximum
- **Field accuracy**: 82-87% minimum
- **Lab accuracy**: 99%+ target
- **Unknown detection**: 94% precision, 91% recall

### Model Constraints
- **Model size**: <52MB compressed (Core ML)
- **Memory usage**: <300MB peak
- **Battery usage**: 3.2% per hour continuous use
- **Thermal threshold**: Degrade gracefully at >45°C

### iOS Deployment Constraints
- **Quantization**: Float16 ONLY (not int8 - loses accuracy)
- **Target devices**: iPhone 12+ (A14 Bionic minimum)
- **Neural Engine**: Must utilize for Tier 1 model
- **Frameworks**: Core ML 5.0+ required

## Critical Components (Must Include)

### Preprocessing Pipeline
- **LASSR super-resolution**: 21% accuracy gain (MANDATORY)
- **U-Net segmentation**: 30-40% accuracy gain (MANDATORY)
- **Illumination normalization**: Required for field use
- **Total preprocessing**: 400-600ms budget

### Data Pipeline
- **CycleGAN augmentation**: Prevents 45-68% field accuracy drop (CRITICAL)
- **Group-aware splitting**: No plant/location leakage (MANDATORY)
- **HEIC support**: Native iPhone format (REQUIRED)

### Model Architecture
- **Three-tier cascade**: Required for speed/accuracy balance
- **Uncertainty quantification**: Three methods minimum
- **Ensemble for Tier 3**: Multiple model consensus

## Do NOT (Forbidden Practices)

### Data Handling
- ❌ Mix training/test data from same plants
- ❌ Use same location images across splits
- ❌ Ignore temporal relationships in splitting
- ❌ Skip augmentation for field deployment

### Model Development
- ❌ Use int8 quantization (degrades accuracy)
- ❌ Skip preprocessing steps to save time
- ❌ Deploy without uncertainty quantification
- ❌ Ignore thermal management

### Optimization
- ❌ Over-prune (>25% sparsity degrades accuracy)
- ❌ Use aggressive quantization
- ❌ Optimize for lab only (field is critical)
- ❌ Ignore battery/thermal constraints

## Acceptable Trade-offs

### Can Sacrifice
- Lab accuracy (99.9% → 99%) for better field performance
- Model size (40MB → 52MB) for accuracy
- Tier 1 accuracy (95% → 93%) for speed
- Training time for better augmentation

### Cannot Sacrifice
- Field accuracy below 82%
- Inference time above 2 seconds
- Unknown detection performance
- Preprocessing pipeline components

## Platform-Specific Constraints

### iPhone Hardware
- A14 Bionic chip minimum (iPhone 12+)
- 6GB RAM assumption
- Neural Engine v4+
- Metal Performance Shaders

### Software Requirements
- iOS 15.0 minimum
- Core ML 5.0+
- Swift 5.5+
- Metal 3.0+

## Environmental Constraints

### Field Conditions
- Direct sunlight: Must handle overexposure
- Low light: Must handle underexposure
- Wet conditions: Must handle reflections
- Motion blur: Must handle handheld capture

### Geographic Variations
- Disease patterns across regions
- Different environmental conditions
- Varied climate zones
- Seasonal changes

## Validation Constraints

### Testing Requirements
- Minimum 100 field deployments
- All lighting conditions tested
- All disease stages validated
- Edge cases documented

### Performance Metrics
- Report stratified by condition
- Confidence calibration required
- Uncertainty validation mandatory
- Battery/thermal monitoring

## Development Constraints

### Code Quality
- Type hints required (Python)
- Documentation for all functions
- Unit tests for critical paths
- Integration tests for pipeline

### Version Control
- No large models in git
- Use git-lfs for datasets
- Document all experiments
- Tag stable releases

## Legal/Ethical Constraints

### Data Privacy
- No user data collection without consent
- Local processing only (no cloud)
- GDPR/CCPA compliant
- No location tracking

### Model Bias
- Test across ethnicities' farming practices
- Validate on diverse crop varieties
- Document limitations clearly
- Provide uncertainty estimates

## Claude Code Freedom

Within these constraints, Claude Code has freedom to:
- Choose specific implementations
- Select optimal libraries
- Design architecture details
- Tune hyperparameters
- Create novel solutions

As long as all hard requirements are met