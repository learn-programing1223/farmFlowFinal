# Retinex Illumination Normalization

## Purpose
- Handles 45-68% field accuracy drop
- Separates illumination from reflectance

## Requirements
- 150-200ms processing time
- LAB color space conversion
- CLAHE with clipLimit=3.0

## Impact
- Reduces lighting variation from 35% to 8%