import os
import json
from pathlib import Path
from tqdm import tqdm
from data import MaskGenerator, MaskGenerationConfig
from config import load_config
from PIL import Image
import numpy as np

config = load_config('config/config.yaml')

# Initialize mask generator
mask_config = MaskGenerationConfig(
    method=config.get('dataset_generation.mask_method'),
    template_threshold=config.get('dataset_generation.template_matching_threshold'),
    dilation_kernel_size=config.get('dataset_generation.mask_dilation_kernel')
)

mask_generator = MaskGenerator(
    character_set=config.get('dataset_generation.character_set'),
    image_size=tuple(config.get('dataset_generation.image_size')),
    config=mask_config
)

# Find missing masks
data_dir = Path(config.get('paths.synthetic_data_dir'))
images_dir = data_dir / "images"
masks_dir = data_dir / "masks"
metadata_dir = data_dir / "metadata"

# Check which masks are missing
missing = []
for i in range(10000):
    sample_id = f"sample_{i:06d}"
    mask_path = masks_dir / f"{sample_id}_mask.png"
    if not mask_path.exists():
        missing.append(sample_id)

print(f"Found {len(missing)} missing masks")

# Generate missing masks
for sample_id in tqdm(missing, desc="Generating missing masks"):
    try:
        # Load metadata
        metadata_path = metadata_dir / f"{sample_id}.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load image
        image_path = images_dir / f"{sample_id}.png"
        image = Image.open(image_path)
        image_array = np.array(image)
        
        # Generate mask
        mask = mask_generator.generate_mask_from_annotations(
            image=image_array,
            annotations=metadata['annotations'],
            font_path=str(Path("fonts") / f"{metadata['font_name']}.ttf"),
            font_size=metadata['font_size']
        )
        
        # Save mask
        mask_path = masks_dir / f"{sample_id}_mask.png"
        mask_generator.save_mask(mask, str(mask_path))
        
    except Exception as e:
        print(f"Error processing {sample_id}: {e}")
        continue

print(f"Completed! Total masks: {len(list(masks_dir.glob('*.png')))}")