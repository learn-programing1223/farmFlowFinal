# Hybrid CNN-ViT Ensemble (Tier 3 Model)

## Disease Pattern Focus

Model should learn to detect:
- Disease visual symptoms (NOT plant features)
- Cross-species disease patterns
- Texture abnormalities
- Color deviations from healthy
- Lesion patterns and progression

The same disease classifier works on ANY plant:
- Powdery mildew (roses, cucumbers, grapes, squash, etc.)
- Blight (tomatoes, potatoes, peppers, eggplants, etc.)
- Mosaic virus (ANY affected plant)
- ALL diseases detected universally across species

## Role in Cascade
- Third tier: Maximum accuracy
- Used for most complex/ambiguous cases
- Final decision maker when lower tiers uncertain

## Performance Requirements (from research)
- Inference time: 1.2-1.5s total
- Accuracy: Highest possible (approaching 99.97%)
- Ensemble of 3-5 models
- Total size: ~50MB after optimization

## Ensemble Composition (from research)
Optimal combination includes:
1. EfficientNet variant (texture features)
2. PMVT - Plant Mobile Vision Transformer (attention-based)
3. Lightweight ConvNeXt (hierarchical patterns)

## Ensemble Strategy Options
- Weighted voting (4.2% improvement over equal weights)
- Stacking with meta-learner (98.05% validation accuracy)
- Dynamic weighting based on confidence

## Key Requirements
- 2-8% accuracy improvement over single models
- Genetic algorithm optimization for weights
- Per-class weight optimization
- Cross-validation for meta-learner

## Test-Time Augmentation (TTA)
- 12-15 augmentations feasible in time budget
- 2-3% additional accuracy gain
- Confidence-weighted aggregation
- 600-900ms additional processing

## When This Ensemble is Used
- Confidence < 70% from Tier 2
- Critical disease identification needed
- Multiple diseases suspected
- User requests highest accuracy mode
- Research/diagnostic applications

## Integration Requirements
- Must provide uncertainty quantification
- All models must be Core ML compatible
- Progressive loading for memory management
- Thermal-aware (can reduce ensemble size if hot)

## Memory Management
- Load on-demand (not kept in memory)
- Can load models progressively
- Must stay under 300MB total memory

Related:
- Research document: Ensemble Strategies section
- Individual models: efficientnet.md, efficientformer.md
- Inference cascade: /inference/pipeline.md
- TTA strategy: /inference/tta.md