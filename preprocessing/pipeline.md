# Preprocessing Pipeline Requirements

## Focus on Disease Features (NOT Plant Features)

Preprocessing should enhance DISEASE SYMPTOMS:
- Lesions, spots, discoloration
- Texture changes (powdery, fuzzy)
- Color abnormalities (yellowing, browning)
- Pattern irregularities

NOT focusing on:
- Plant species identification features
- Leaf shape (unless affected by disease)
- Plant structure (unless diseased)

## CRITICAL PERFORMANCE TARGETS (from research)
- **Total contribution**: 25-35% accuracy improvement
- **Total time budget**: 400-600ms
- **LASSR alone**: 21% accuracy gain (200-400ms)
- **Segmentation**: 30-40% accuracy gain (300-500ms)

## REQUIRED COMPONENTS (in order)
1. **LASSR super-resolution**
   - 21% accuracy improvement
   - 200-400ms processing time
   - Critical for low-resolution field images

2. **Multi-resolution processing**
   - 384x384 optimal balance
   - 512x512 maximum accuracy
   - Dynamic selection based on input quality

3. **Illumination normalization**
   - Retinex-based decomposition
   - Handles 45-68% field accuracy drop
   - 150-200ms processing time

4. **Segmentation**
   - U-Net with 98.66% accuracy
   - 300-500ms processing time
   - Attention mechanisms for boundaries

5. **Color space fusion**
   - RGB, LAB, HSV, YCbCr
   - Extract disease-specific features
   - Parallel processing required

6. **Vegetation indices**
   - VARI (Visible Atmospherically Resistant Index)
   - MGRVI (Modified Green Red Vegetation Index)
   - vNDVI (visible Normalized Difference Vegetation Index)

## CONSTRAINTS
- Must handle iPhone HEIC format natively
- Must preserve disease features during enhancement
- Must work with variable lighting conditions (direct sun to shade)
- Cannot exceed 600ms total processing time
- Must maintain accuracy gains across all disease categories

## VALIDATION METRICS
- Measure accuracy with/without each component
- Time each component separately
- Validate on field images specifically
- Check robustness to lighting variations

## Related Research
- See research document sections on preprocessing
- IBM paper on preprocessing impact
- LASSR super-resolution methodology

## Implementation Notes
Let Claude Code determine optimal implementation based on these constraints