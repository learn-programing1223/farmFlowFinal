# U-Net Segmentation

## Performance Targets
- 98.66% segmentation accuracy
- 30-40% overall accuracy improvement
- 300-500ms inference time

## Cascade Strategy
1. RGB thresholding (10-20ms)
2. GrabCut (100-150ms)
3. U-Net (300-500ms)
4. Morphological cleanup (20-30ms)

## Fallback
- Center crop if segmentation fails