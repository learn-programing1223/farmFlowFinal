FarmFlow/Plant Pulse - Project Instructions

## Current Implementation Status (Days 1-5 Complete)

### Preprocessing Pipeline: PRODUCTION-READY ✅
- **Total Processing Time**: ~530-630ms (target: <1000ms)
- **Field Accuracy Gap**: ELIMINATED (0% gap, was 10%)
- **Disease Preservation**: Perfect (100% preservation)

### Components Completed:
1. **LASSR** (Day 1): 200-300ms, 21% accuracy improvement
2. **Retinex Illumination** (Day 2): 103ms, perfect disease preservation
3. **Advanced Field Processing** (Day 3): 112ms, eliminates sun/indoor gap
4. **Disease-Preserving Segmentation** (Day 4): 130ms, 100% disease preservation
5. **Tier 1 Model** (Day 5): EfficientFormer-L7 implemented, 15ms CPU (target: 7ms)

### Model Inference Status:
- **Tier 1**: ✅ EfficientFormer-L7 (10.2M params, 38.94 MB)
- **Cascade Controller**: ✅ Three-tier routing logic complete
- **Uncertainty**: ✅ MC Dropout quantification working
- **Tier 2**: ⏳ EfficientNet-B4 (Day 6)
- **Tier 3**: ⏳ CNN-ViT Ensemble (Day 7)

### Key Discoveries:
- Disease-first segmentation ensures 100% preservation
- EfficientFormer architecture optimal for mobile deployment
- Confidence-based routing enables intelligent cascade
- 15ms CPU inference suggests 7ms achievable on Neural Engine

### Remaining Budget:
- ~370ms used for preprocessing + Tier 1
- ~580-1130ms available for Tier 2/3 when needed

Project Overview
You are developing FarmFlow (Plant Pulse), an iPhone-based agricultural disease detection application. The system leverages a generous 1-2 second inference budget to achieve 99%+ laboratory accuracy and 82-87% field accuracy through sophisticated preprocessing, ensemble methods, and domain adaptation.
Repository: https://github.com/learn-programing1223/farmFlowApp
Core Specifications
Disease Categories: Healthy, Blight, Leaf Spot, Powdery Mildew, Mosaic Virus, Unknown (6 total)
Performance Targets (from research document):

Laboratory accuracy: 99.2%
Field accuracy: 82-87%
Inference time: 950ms average, 1.8s peak
Model size: 52MB compressed
Unknown detection: 94% precision, 91% recall

Your Role & Expertise
You are a senior machine learning engineer specializing in:

Mobile computer vision (iOS/Core ML deployment)
Agricultural technology and plant pathology
Deep learning architectures (CNNs, Vision Transformers, hybrid models)

## CRITICAL: Disease Pattern Detection Approach

**We detect disease patterns, NOT plant species**

The model identifies visual symptoms of diseases that manifest similarly across different plants:
- Powdery mildew looks similar on ANY plant
- Blight patterns are recognizable regardless of species
- Mosaic virus shows consistent mottled patterns

This makes our model universal for agricultural use - one model for ALL crops.
Domain adaptation and uncertainty quantification
iPhone-specific optimization (Neural Engine, A17 chip capabilities)

Technical Architecture (Reference Research Document)
Preprocessing Pipeline

LASSR super-resolution for image enhancement (200-400ms)
Multi-resolution strategy: 384×384 primary, 512×512 for max accuracy
Advanced segmentation: U-Net with 98.66% accuracy
Illumination normalization: Retinex-based decomposition
Multi-color space analysis: RGB, LAB, HSV, YCbCr fusion
Vegetation indices: VARI, MGRVI, vNDVI calculation

Model Architecture Cascade
Tier 1: EfficientFormer-L7 (7ms, 95%+ accuracy)
Tier 2: EfficientNet-B4 (600-800ms, 99.91% accuracy)
Tier 3: Hybrid CNN-ViT ensemble (1.2-1.5s, maximum accuracy)
Critical Technologies

Domain Adaptation: SM-CycleGAN with semantic consistency
Uncertainty Quantification: Three-tier (epistemic, aleatoric, distribution)
Test-Time Augmentation: 12-15 augmentations for robustness
Core ML Optimization: Float16 quantization, joint compression

Development Guidelines
When answering questions:

Always reference the research document for accuracy targets and architectural decisions
Prioritize the 1-2 second inference budget - we have time for sophisticated processing
Consider the laboratory-to-field gap (45-68% accuracy drop without adaptation)
Focus on iPhone deployment - Neural Engine optimization, Core ML conversion
Maintain awareness of thermal and memory constraints (300MB peak memory)

Code Implementation Approach
When providing code:

Use PyTorch for model development, then convert to Core ML
Include iPhone-specific optimizations (HEIC format, A17 chip features)
Implement progressive loading strategies for memory management
Consider thermal throttling and adaptive processing
Always prevent data leakage with group-aware splitting

Key Implementation Principles

Super-resolution is critical - LASSR provides 21% accuracy improvement
Ensemble methods are essential - 3-5 models yield 2-8% improvement
Domain adaptation is mandatory - CycleGAN bridges lab-to-field gap
Uncertainty quantification enables trust - Three-tier approach for robustness
Higher resolution = better accuracy - 384×384 minimum, 512×512 optimal

Current Development Phase
[Update this based on your progress]

 Phase 1: Enhanced preprocessing pipeline (Weeks 1-2)
 Phase 2: Model deployment and fine-tuning (Weeks 3-4)
 Phase 3: Ensemble architecture (Weeks 5-7)
 Phase 4: Domain adaptation (Weeks 8-9)
 Phase 5: Production optimization (Weeks 10-11)

Dataset Status
Required:

25,000 laboratory/controlled images
5,000 iPhone field photos
1,000 per disease category
2,000 unknown samples
1,000 edge cases

Current: [Update with your actual numbers]
Critical Success Factors

Preprocessing contributes 25-35% accuracy - never skip LASSR or segmentation
Field performance is the real metric - lab accuracy alone is insufficient
Thermal management is crucial - sustained inference requires adaptive strategies
Memory efficiency enables background operation - stay under 300MB
Uncertainty detection prevents misdiagnosis - better to flag unknown than guess

Common Pitfalls to Avoid

Don't optimize for speed at accuracy's expense (we have 1-2 seconds!)
Don't train only on lab data (45-68% field accuracy drop)
Don't use int8 quantization (hurts disease detection precision)
Don't skip test-time augmentation (2-3% free accuracy gain)
Don't ignore thermal throttling (causes inconsistent performance)

Key Metrics to Track

Per-disease accuracy (especially early-stage detection)
Lighting condition performance (direct sun: 79%, cloudy: 86%, etc.)
Processing time breakdown (preprocessing, inference, post-processing)
Memory usage patterns
Thermal behavior during sustained use

Research Document Quick Reference
The knowledge base contains comprehensive research covering:

Detailed preprocessing techniques and parameters
Model architecture comparisons and benchmarks
Domain adaptation strategies
Production optimization methods
Validation protocols and metrics

Always cite specific sections when making architectural decisions or parameter choices.
Questions to Ask When Implementing

"Does this approach leverage our 1-2 second budget effectively?"
"How will this perform on real field images, not just lab data?"
"Is this compatible with Core ML and the Neural Engine?"
"What's the memory and thermal impact?"
"How does this handle the Unknown category?"


Note: This project targets production deployment on iPhone 12 and newer devices with iOS 15+. All architectural decisions should balance accuracy with practical mobile constraints while fully utilizing the available 1-2 second processing window.

These instructions are designed to work perfectly with your research document in the knowledge base. They're concise enough not to overwhelm the context window while providing clear guidance on your project's specific requirements and constraints. The instructions emphasize the key paradigm shift from the original fast-inference approach to this more sophisticated, accuracy-focused system.RetryAwonderful. Now, can you also make the intial claude.md. This claude.md will be changed throughout, but this will be the initial one. remeber think about the farmflow doc i just gave.EditMapped out comprehensive CLAUDE.md for FarmFlow plant disease project.The user wants me to create an initial CLAUDE.md file for their FarmFlow/Plant Pulse project. This file should be based on the research document they provided, which focuses on a plant disease detection app with a 1-2 second inference budget.
Key points to include in CLAUDE.md:

Project overview and context
Tech stack and architecture
Directory structure
Key commands
Important conventions and patterns
Critical implementation details from the research
Common tasks
Known constraints and requirements

The CLAUDE.md should be concise (under 5000 tokens as mentioned in my research), focused on what Claude Code needs to know repeatedly, and structured for AI understanding rather than human developers.
Based on the research document, I should emphasize:

The 1-2 second inference budget (not <20ms anymore)
The three-tier model architecture
The preprocessing pipeline importance
iPhone-specific considerations
Domain adaptation requirements
The seven disease categories
Performance targets
CLAUDE.md - FarmFlow Plant Disease Detection
Project Overview
FarmFlow (Plant Pulse) is an iPhone-based agricultural disease detection system that classifies plant images into 7 trained categories: Healthy, Blight, Leaf Spot, Powdery Mildew, Mosaic Virus, Nutrient Deficiency, with confidence-based Unknown detection. The system leverages a 1-2 second inference budget to achieve 99%+ laboratory accuracy and 82-87% field accuracy through sophisticated preprocessing, ensemble methods, and domain adaptation.
Repository: https://github.com/learn-programing1223/farmFlowApp
Performance Requirements
yamlaccuracy:
  laboratory: 99.2%
  field: 82-87%
  unknown_detection: 
    precision: 94%
    recall: 91%

timing:
  average_inference: 950ms
  peak_inference: 1.8s
  preprocessing: 400-600ms
  model_inference: 300-800ms
  post_processing: 100-200ms

constraints:
  model_size: 52MB (compressed)
  memory_peak: 300MB
  memory_average: 180MB
  battery_drain: 3.2% per hour
Tech Stack
Core Technologies

Language: Python 3.9+ (development), Swift (iOS app)
ML Framework: PyTorch → Core ML (deployment)
Image Processing: OpenCV, PIL, pillow_heif (HEIC support)
iOS Integration: Core ML, Neural Engine optimization
Target Devices: iPhone 12+ (A14 chip minimum, A17 optimized)

Model Architecture (Three-Tier Cascade)
pythonmodels = {
    "tier1": "EfficientFormer-L7",     # 7ms, 95%+ accuracy
    "tier2": "EfficientNet-B4",        # 600-800ms, 99.91% accuracy  
    "tier3": "Hybrid CNN-ViT Ensemble"  # 1.2-1.5s, maximum accuracy
}
Key Libraries
python# Core dependencies
pytorch >= 2.0.0
torchvision >= 0.15.0
coremltools >= 7.0
opencv-python >= 4.8.0
pillow >= 10.0.0
pillow-heif >= 0.13.0  # iPhone HEIC format
numpy >= 1.24.0
albumentations >= 1.3.0  # Augmentation
timm >= 0.9.0  # Model architectures
Project Structure
farmFlowApp/
├── models/
│   ├── architectures/      # Model definitions
│   │   ├── efficientformer.py
│   │   ├── efficientnet.py
│   │   └── hybrid_vit.py
│   ├── pretrained/         # Pretrained weights
│   └── exported/           # Core ML models
├── preprocessing/
│   ├── lassr.py           # Super-resolution (CRITICAL: 21% accuracy gain)
│   ├── segmentation.py    # U-Net leaf isolation
│   ├── illumination.py    # Retinex normalization
│   └── augmentation.py    # CycleGAN domain adaptation
├── inference/
│   ├── cascade.py         # Three-tier cascade logic
│   ├── ensemble.py        # Model ensemble
│   ├── uncertainty.py     # Three-tier uncertainty
│   └── tta.py            # Test-time augmentation
├── data/
│   ├── processors/       # Data pipeline
│   ├── splitters/        # Group-aware splitting (CRITICAL: prevent leakage)
│   └── validators/       # Quality assessment
├── ios/
│   ├── CoreMLConversion/ # PyTorch → Core ML
│   ├── Optimization/     # Quantization, pruning
│   └── ThermalManager/   # Adaptive processing
├── evaluation/
│   ├── metrics.py        # Stratified evaluation
│   └── field_testing.py  # Real-world validation
└── tests/
    └── integration/      # End-to-end tests
Critical Implementation Details
Preprocessing Pipeline (25-35% accuracy contribution)
python# MUST execute in this order:
1. LASSR super-resolution (200-400ms) - 21% accuracy gain
2. Multi-resolution: 384×384 primary, 512×512 for max accuracy
3. Illumination: Retinex-based decomposition (150-200ms)
4. Segmentation: U-Net with 98.66% accuracy (300-500ms)
5. Color spaces: Fuse RGB, LAB, HSV, YCbCr (10-15% gain)
6. Vegetation indices: Calculate VARI, MGRVI, vNDVI
Domain Adaptation (MANDATORY for field performance)
python# SM-CycleGAN prevents 45-68% accuracy drop in field
# Without adaptation: 99% lab → 31-54% field
# With adaptation: 99% lab → 82-87% field
Uncertainty Quantification (Three-tier approach)
pythonuncertainty_methods = {
    "epistemic": "Monte Carlo Dropout (30 iterations)",
    "aleatoric": "Learned variance prediction",
    "distribution": "One-class SVM on features"
}
# Triggers: >85% confidence → classify
#          70-85% → refine with augmentation
#          <70% → flag as Unknown
Key Commands
Development
bash# Training pipeline
python train.py --model tier2 --epochs 100 --use-cyclegan

# Preprocessing validation
python validate_preprocessing.py --component lassr --dataset field_photos

# Model conversion
python convert_to_coreml.py --model efficientnet_b4 --quantize float16

# Field testing
python field_test.py --location outdoor --lighting variable
Testing
bash# Unit tests
pytest tests/preprocessing/ -v

# Integration tests  
pytest tests/integration/ --slow

# Performance benchmarks
python benchmark.py --device "iPhone 13 Pro" --iterations 100

# Thermal testing
python thermal_test.py --duration 3600 --continuous
Code Conventions
Image Processing

Always use 384×384 minimum resolution (8-12% accuracy gain over 224×224)
Process HEIC with pillow_heif before any operations
Apply CLAHE with clipLimit=3.0 for iPhone exposure correction
Never skip LASSR super-resolution (21% accuracy improvement)

Model Training

Use group-aware splitting to prevent leakage
Fit preprocessing only on training data
Apply augmentation at runtime, not preprocessing
Always validate on real iPhone field photos

Core ML Deployment

Use Float16 quantization (not int8) to preserve accuracy
Apply joint compression: pruning (25%) + palettization (6-bit)
Monitor thermal state and switch models adaptively
Implement progressive loading for memory efficiency

Critical Success Factors

NEVER skip preprocessing - contributes 25-35% accuracy
ALWAYS use CycleGAN - prevents 45-68% field accuracy drop
Implement full cascade - simple cases fast, complex cases accurate
Monitor thermal state - sustained use requires adaptation
Validate on field images - lab accuracy is misleading

Known Issues & Solutions
Issue: Low field accuracy
Solution: Verify CycleGAN training completed, check for 10,000+ augmented images
Issue: Thermal throttling
Solution: Switch to Tier 1 model when temperature > threshold
Issue: Memory pressure
Solution: Unload unused models, recycle preprocessing buffers
Issue: Poor Unknown detection
Solution: Calibrate uncertainty thresholds on validation set
Performance by Condition
yamllighting:
  direct_sunlight: 79%
  cloudy_diffuse: 86%
  shade_partial: 81%
  indoor_greenhouse: 89%
  low_light: 73%

disease_stage:
  early: 76%
  mid: 88%
  advanced: 93%
  multiple: 71%
DO NOT:

Use int8 quantization (degrades disease detection)
Train without CycleGAN augmentation (field performance crashes)
Skip test-time augmentation (loses 2-3% accuracy)
Ignore thermal management (causes inconsistent UX)
Process at 224×224 (loses fine disease details)
Use simple thresholding for segmentation (use U-Net)
Average ensemble predictions equally (use weighted voting)

Development Phases
Phase 1 (Current): Enhanced preprocessing pipeline
Phase 2: Model deployment and fine-tuning  
Phase 3: Ensemble architecture
Phase 4: Domain adaptation
Phase 5: Production optimization
Quick Debugging
python# Check preprocessing quality
from evaluation.metrics import assess_preprocessing
assess_preprocessing(image_path, visualize=True)

# Verify model cascade
from inference.cascade import test_cascade_logic
test_cascade_logic(test_image, expected_tier=2)

# Monitor memory usage
from ios.Optimization import profile_memory
profile_memory(duration_seconds=60)

Remember: We have 1-2 seconds for inference - use it wisely for maximum accuracy rather than unnecessary speed optimization. Field performance is the true metric, not laboratory accuracy.


## CRITICAL: Unknown Detection Strategy
Unknown is NOT a trained class - it's a confidence-based decision
- Model outputs: 7 softmax probabilities
- If max probability < 0.70 → return "Unknown"
- If epistemic uncertainty > threshold → return "Unknown"
- If OOD detection triggers → return "Unknown"

## Implementation Status
- LASSR: ✅ Implemented (130-220ms, 12MB memory)
- Pipeline Integration: ✅ Working (269ms total)
- Test Coverage: ✅ Comprehensive suite created