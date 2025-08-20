# Core ML Conversion Requirements

## Specifications
- Float16 quantization (NOT int8)
- 25% weight pruning
- 6-bit palettization
- Target: <52MB model size

## Memory Constraints
- Peak: <300MB
- Average: 180MB

## Performance
- Maintain <1% accuracy loss
- 4-6× size reduction