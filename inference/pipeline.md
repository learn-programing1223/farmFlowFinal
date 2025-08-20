# Three-Tier Cascade Pipeline

## Inference Approach

When processing an image:
1. Look for disease symptoms (spots, discoloration, texture)
2. Classify based on disease pattern
3. DO NOT consider plant species

Output: "Powdery Mildew detected" 
NOT: "Powdery Mildew on roses detected"

## Tier 1: EfficientFormer-L7
- 7ms inference
- For high-confidence cases

## Tier 2: EfficientNet-B4
- 600-800ms inference
- When Tier 1 confidence < 85%

## Tier 3: Ensemble
- 1.2-1.5s inference
- For complex/ambiguous cases

## Unknown Detection
- Triggered when confidence < 70%
- Based on uncertainty metrics from /inference/uncertainty.md