# Preprocessing Module

## Requirements from Research

### Performance Targets
- **Total accuracy gain**: 25-35%
- **Processing time**: 400-600ms total
- **LASSR contribution**: 21% accuracy improvement
- **Segmentation contribution**: 30-40% accuracy improvement

### Component Timing Budget
| Component | Time Budget | Accuracy Contribution |
|-----------|------------|----------------------|
| LASSR Super-resolution | 200-400ms | 21% |
| U-Net Segmentation | 300-500ms | 30-40% |
| Illumination Normalization | 150-200ms | 8-12% |
| Color Space Fusion | 50-100ms | 5-8% |
| Vegetation Indices | 20-50ms | 3-5% |

## Components Needed

### 1. LASSR Super-resolution
- Lightweight Attention-based Super-Resolution
- 2x upscaling for low-quality field images
- Preserves disease features while enhancing clarity
- Critical for iPhone field photos

### 2. U-Net Segmentation
- 98.66% segmentation accuracy requirement
- Attention mechanisms for leaf boundaries
- Fallback strategies for failed segmentation
- Must handle overlapping leaves

### 3. Retinex Illumination Normalization
- Separates illumination from reflectance
- Handles direct sunlight to deep shade
- Reduces lighting-induced variance from 35% to 8%
- Critical for field deployment

### 4. Multi-color Space Fusion
- RGB: Original color information
- LAB: Perceptual color differences
- HSV: Hue-based disease detection
- YCbCr: Luminance separation

### 5. Vegetation Indices Calculation
- VARI: (Green - Red) / (Green + Red - Blue)
- MGRVI: (Green² - Red²) / (Green² + Red²)
- vNDVI: (NIR - Red) / (NIR + Red) approximation

## Constraints

### Format Requirements
- Handle iPhone HEIC format natively
- Support JPEG, PNG fallbacks
- Preserve EXIF metadata for GPS/timestamp

### Quality Preservation
- Maintain disease feature visibility
- Avoid over-smoothing diseased areas
- Preserve color accuracy for diagnosis

### Robustness Requirements
- Work with motion blur (handheld capture)
- Handle partial occlusion
- Process wet leaves (rain/dew)
- Adapt to variable lighting

## Directory Structure
```
preprocessing/
├── README.md (this file)
├── pipeline.md (specifications)
├── lassr.py (to implement)
├── segmentation/
│   ├── unet.py (to implement)
│   └── fallback.py (to implement)
├── illumination/
│   ├── retinex.py (to implement)
│   └── adaptive.py (to implement)
├── color/
│   ├── fusion.py (to implement)
│   └── indices.py (to implement)
└── tests/
    └── validate_gains.py (to implement)
```

## Validation Requirements

### Accuracy Validation
- Test with/without each component
- Measure cumulative gains
- Validate on field images specifically

### Performance Validation
- Profile each component separately
- Measure memory usage
- Test on iPhone hardware

### Robustness Testing
- Variable lighting conditions
- Different disease stages
- Disease features across various leaves
- Edge cases (blur, occlusion)

## Integration Points
- Data pipeline: `/data/loaders.py`
- Model input: `/models/architectures/`
- iOS optimization: `/ios/Optimization/`

## Let Claude Code Determine Implementation
The specific implementation details should be optimized based on:
- Available libraries and frameworks
- Memory and compute constraints
- Actual performance on test data
- Integration with existing pipeline