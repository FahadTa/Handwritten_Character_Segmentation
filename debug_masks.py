import json
import numpy as np
from PIL import Image
from pathlib import Path

# Check sample_000000
print("="*60)
print("DIAGNOSTIC FOR SAMPLE_000000")
print("="*60)

# 1. Load metadata
with open('outputs/synthetic_data/metadata/sample_000000.json', 'r') as f:
    metadata = json.load(f)

print(f"\n1. METADATA:")
print(f"   Total annotations: {len(metadata['annotations'])}")

# Count spaces vs visible
spaces = sum(1 for ann in metadata['annotations'] if ann['character'] == ' ')
visible = len(metadata['annotations']) - spaces

print(f"   Space characters: {spaces}")
print(f"   Visible characters: {visible}")
print(f"   Text: {metadata['text_content']}")

# 2. Load mask
mask = np.array(Image.open('outputs/synthetic_data/masks/sample_000000_mask.png'))

print(f"\n2. MASK:")
print(f"   Mask shape: {mask.shape}")
print(f"   Mask dtype: {mask.dtype}")
print(f"   Mask min value: {mask.min()}")
print(f"   Mask max value: {mask.max()}")

# Count unique classes
unique_classes = np.unique(mask)
print(f"   Unique class IDs in mask: {len(unique_classes)}")
print(f"   Class IDs: {unique_classes[:20]}...")  # First 20

# Count non-background pixels
non_bg = np.sum(mask > 0)
bg = np.sum(mask == 0)
print(f"   Background pixels: {bg}")
print(f"   Character pixels: {non_bg}")

# Check which characters are in mask
chars_in_mask = set(unique_classes) - {0}
print(f"\n3. CHARACTERS DETECTED:")
print(f"   Number of character classes found: {len(chars_in_mask)}")
print(f"   Class IDs found: {sorted(chars_in_mask)}")

print("\n" + "="*60)