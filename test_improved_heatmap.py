"""
Test improved disease detector - should show selective detection
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from preprocessing.segmentation.disease_detector import DiseaseRegionDetector
from preprocessing.illumination.disease_pattern_generator import DiseasePatternGenerator

print("Testing Improved Disease Detector")
print("=" * 50)

# Initialize components
generator = DiseasePatternGenerator()
detector = DiseaseRegionDetector(sensitivity=0.5)  # Medium sensitivity

# Test 1: Healthy leaf (should show minimal detection)
print("\n1. Testing HEALTHY leaf...")
healthy_leaf = generator.create_healthy_leaf((384, 384))
healthy_mask, healthy_info = detector.detect(healthy_leaf)
print(f"   Healthy leaf coverage: {healthy_info['total_coverage']:.1%}")
print(f"   Expected: <5%, Actual: {healthy_info['total_coverage']:.1%}")

# Test 2: Mild disease (should show 10-20% coverage)
print("\n2. Testing MILD disease...")
mild_diseased = generator.create_blight_pattern(healthy_leaf, severity='mild')
mild_mask, mild_info = detector.detect(mild_diseased.image)
print(f"   Mild disease coverage: {mild_info['total_coverage']:.1%}")
print(f"   Expected: 10-20%, Actual: {mild_info['total_coverage']:.1%}")

# Test 3: Moderate disease (should show 20-40% coverage)
print("\n3. Testing MODERATE disease...")
moderate_diseased = generator.create_blight_pattern(healthy_leaf, severity='moderate')
moderate_mask, moderate_info = detector.detect(moderate_diseased.image)
print(f"   Moderate disease coverage: {moderate_info['total_coverage']:.1%}")
print(f"   Expected: 20-40%, Actual: {moderate_info['total_coverage']:.1%}")

# Test 4: Severe disease (should show 40-60% coverage)
print("\n4. Testing SEVERE disease...")
severe_diseased = generator.create_blight_pattern(healthy_leaf, severity='severe')
severe_mask, severe_info = detector.detect(severe_diseased.image)
print(f"   Severe disease coverage: {severe_info['total_coverage']:.1%}")
print(f"   Expected: 40-60%, Actual: {severe_info['total_coverage']:.1%}")

# Create visualization
fig, axes = plt.subplots(4, 3, figsize=(12, 16))

# Row 1: Healthy
axes[0, 0].imshow(healthy_leaf)
axes[0, 0].set_title('Healthy Leaf')
axes[0, 0].axis('off')

axes[0, 1].imshow(healthy_mask, cmap='hot', vmin=0, vmax=255)
axes[0, 1].set_title(f'Heatmap ({healthy_info["total_coverage"]:.1%})')
axes[0, 1].axis('off')

axes[0, 2].imshow(healthy_leaf)
axes[0, 2].imshow(healthy_mask, cmap='hot', alpha=0.5, vmin=0, vmax=255)
axes[0, 2].set_title('Overlay')
axes[0, 2].axis('off')

# Row 2: Mild
axes[1, 0].imshow(mild_diseased.image)
axes[1, 0].set_title('Mild Disease')
axes[1, 0].axis('off')

axes[1, 1].imshow(mild_mask, cmap='hot', vmin=0, vmax=255)
axes[1, 1].set_title(f'Heatmap ({mild_info["total_coverage"]:.1%})')
axes[1, 1].axis('off')

axes[1, 2].imshow(mild_diseased.image)
axes[1, 2].imshow(mild_mask, cmap='hot', alpha=0.5, vmin=0, vmax=255)
axes[1, 2].set_title('Overlay')
axes[1, 2].axis('off')

# Row 3: Moderate
axes[2, 0].imshow(moderate_diseased.image)
axes[2, 0].set_title('Moderate Disease')
axes[2, 0].axis('off')

axes[2, 1].imshow(moderate_mask, cmap='hot', vmin=0, vmax=255)
axes[2, 1].set_title(f'Heatmap ({moderate_info["total_coverage"]:.1%})')
axes[2, 1].axis('off')

axes[2, 2].imshow(moderate_diseased.image)
axes[2, 2].imshow(moderate_mask, cmap='hot', alpha=0.5, vmin=0, vmax=255)
axes[2, 2].set_title('Overlay')
axes[2, 2].axis('off')

# Row 4: Severe
axes[3, 0].imshow(severe_diseased.image)
axes[3, 0].set_title('Severe Disease')
axes[3, 0].axis('off')

axes[3, 1].imshow(severe_mask, cmap='hot', vmin=0, vmax=255)
axes[3, 1].set_title(f'Heatmap ({severe_info["total_coverage"]:.1%})')
axes[3, 1].axis('off')

axes[3, 2].imshow(severe_diseased.image)
axes[3, 2].imshow(severe_mask, cmap='hot', alpha=0.5, vmin=0, vmax=255)
axes[3, 2].set_title('Overlay')
axes[3, 2].axis('off')

plt.suptitle('Improved Disease Detection - Selective Heatmaps', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('improved_heatmap_test.png', dpi=100, bbox_inches='tight')
print(f"\n[SUCCESS] Results saved to improved_heatmap_test.png")

# Summary
print("\n" + "=" * 50)
print("SUMMARY:")
print("-" * 50)

success = True
if healthy_info['total_coverage'] > 0.05:
    print(f"[WARN]  Healthy detection too high: {healthy_info['total_coverage']:.1%}")
    success = False
else:
    print(f"[OK] Healthy detection OK: {healthy_info['total_coverage']:.1%}")

if mild_info['total_coverage'] < 0.05 or mild_info['total_coverage'] > 0.30:
    print(f"[WARN]  Mild detection out of range: {mild_info['total_coverage']:.1%}")
    success = False
else:
    print(f"[OK] Mild detection OK: {mild_info['total_coverage']:.1%}")

if moderate_info['total_coverage'] < 0.15 or moderate_info['total_coverage'] > 0.50:
    print(f"[WARN]  Moderate detection out of range: {moderate_info['total_coverage']:.1%}")
    success = False
else:
    print(f"[OK] Moderate detection OK: {moderate_info['total_coverage']:.1%}")

if severe_info['total_coverage'] < 0.30 or severe_info['total_coverage'] > 0.70:
    print(f"[WARN]  Severe detection out of range: {severe_info['total_coverage']:.1%}")
    success = False
else:
    print(f"[OK] Severe detection OK: {severe_info['total_coverage']:.1%}")

print("-" * 50)
if success:
    print("[OK] ALL TESTS PASSED - Disease detector is now selective!")
else:
    print("[WARN]  Some tests need adjustment")

# Test different sensitivities
print("\n" + "=" * 50)
print("SENSITIVITY COMPARISON:")
print("-" * 50)

for sensitivity in [0.3, 0.5, 0.7]:
    det = DiseaseRegionDetector(sensitivity=sensitivity)
    mask, info = det.detect(moderate_diseased.image)
    print(f"Sensitivity {sensitivity}: Coverage = {info['total_coverage']:.1%}")