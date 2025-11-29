"""
Script to regenerate masks using the fixed SEMANTIC segmentation approach.
This will update all existing masks to use character class IDs instead of instance IDs.

Usage:
    # Verify existing masks (no changes)
    python regenerate_masks.py --verify_only
    
    # Regenerate all masks
    python regenerate_masks.py
    
    # Resume from specific sample
    python regenerate_masks.py --start_from 5000
"""

import os
import json
import sys
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from data.mask_generator import MaskGenerator, MaskGenerationConfig
from config import load_config


def regenerate_all_masks(
    data_dir: str,
    config: dict,
    start_from: int = 0,
    verify_only: bool = False
):
    """
    Regenerate all masks with semantic segmentation approach.
    
    Args:
        data_dir: Root directory containing dataset
        config: Configuration dictionary
        start_from: Sample index to start from (for resuming)
        verify_only: Only verify masks, don't regenerate
    """
    data_dir = Path(data_dir)
    images_dir = data_dir / "images"
    masks_dir = data_dir / "masks"
    metadata_dir = data_dir / "metadata"
    
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
    
    print("\n" + "=" * 80)
    print("MASK REGENERATION - SEMANTIC SEGMENTATION")
    print("=" * 80)
    print(f"Data directory: {data_dir}")
    print(f"Number of classes: {mask_generator.num_classes}")
    print(f"Character set size: {len(mask_generator.character_set)}")
    print(f"Mask generation method: {mask_config.method}")
    print(f"Starting from sample: {start_from}")
    print(f"Verify only mode: {verify_only}")
    print("=" * 80 + "\n")
    
    metadata_files = sorted(list(metadata_dir.glob("sample_*.json")))
    
    if start_from > 0:
        metadata_files = metadata_files[start_from:]
        print(f"Resuming from sample {start_from}")
    
    print(f"Total samples to process: {len(metadata_files)}\n")
    
    stats = {
        'total_processed': 0,
        'successful': 0,
        'failed': 0,
        'class_mismatch': 0,
        'max_class_id': 0,
        'class_distribution': {}
    }
    
    for metadata_path in tqdm(metadata_files, desc="Processing masks"):
        try:
            sample_id = metadata_path.stem
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            image_path = images_dir / f"{sample_id}.png"
            if not image_path.exists():
                print(f"\nWarning: Image not found for {sample_id}")
                stats['failed'] += 1
                continue
            
            image = Image.open(image_path)
            image_array = np.array(image)
            
            mask_path = masks_dir / f"{sample_id}_mask.png"
            
            if verify_only:
                if mask_path.exists():
                    mask = mask_generator.load_mask(str(mask_path))
                    max_class = mask.max()
                    
                    if max_class >= mask_generator.num_classes:
                        stats['class_mismatch'] += 1
                        print(f"\nERROR: {sample_id} has class ID {max_class} >= {mask_generator.num_classes}")
                    
                    stats['max_class_id'] = max(stats['max_class_id'], max_class)
                    stats['successful'] += 1
                else:
                    print(f"\nWarning: Mask not found for {sample_id}")
                    stats['failed'] += 1
            else:
                mask = mask_generator.generate_mask_from_annotations(
                    image=image_array,
                    annotations=metadata['annotations'],
                    font_path=str(Path("fonts") / f"{metadata['font_name']}.ttf"),
                    font_size=metadata['font_size']
                )
                
                max_class = mask.max()
                if max_class >= mask_generator.num_classes:
                    print(f"\nERROR: Generated mask for {sample_id} has invalid class ID {max_class}")
                    stats['failed'] += 1
                    continue
                
                mask_generator.save_mask(mask, str(mask_path))
                
                unique_classes = np.unique(mask)
                for class_id in unique_classes:
                    if class_id > 0:
                        stats['class_distribution'][int(class_id)] = stats['class_distribution'].get(int(class_id), 0) + 1
                
                stats['max_class_id'] = max(stats['max_class_id'], max_class)
                stats['successful'] += 1
            
            stats['total_processed'] += 1
            
        except Exception as e:
            print(f"\nError processing {metadata_path.name}: {e}")
            stats['failed'] += 1
            continue
    
    print("\n" + "=" * 80)
    print("REGENERATION COMPLETE")
    print("=" * 80)
    print(f"Total processed: {stats['total_processed']}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print(f"Max class ID found: {stats['max_class_id']}")
    print(f"Expected max class ID: {mask_generator.num_classes - 1}")
    
    if stats['class_mismatch'] > 0:
        print(f"\nWARNING: {stats['class_mismatch']} masks have class IDs >= num_classes!")
        print("This will cause training to fail!")
    else:
        print("\n✅ All masks have valid class IDs")
    
    if not verify_only and stats['class_distribution']:
        print(f"\nCharacter class distribution (top 10):")
        sorted_classes = sorted(stats['class_distribution'].items(), key=lambda x: x[1], reverse=True)
        for class_id, count in sorted_classes[:10]:
            char = [k for k, v in mask_generator.char_to_id.items() if v == class_id]
            char_str = char[0] if char else '?'
            print(f"  Class {class_id} ('{char_str}'): {count} occurrences")
    
    print("=" * 80 + "\n")
    
    return stats


def verify_dataset_integrity(data_dir: str, config: dict):
    """
    Verify that all masks are compatible with the model.
    
    Args:
        data_dir: Dataset directory
        config: Configuration dictionary
    """
    print("\n" + "=" * 80)
    print("VERIFYING DATASET INTEGRITY")
    print("=" * 80)
    
    stats = regenerate_all_masks(data_dir, config, verify_only=True)
    
    num_classes = config.get('model.unet.num_classes')
    
    if stats['max_class_id'] >= num_classes:
        print(f"\n❌ VERIFICATION FAILED!")
        print(f"   Found class IDs up to {stats['max_class_id']}")
        print(f"   But model only supports {num_classes} classes (0-{num_classes-1})")
        print(f"\n   ACTION REQUIRED: Regenerate masks with correct approach")
        return False
    else:
        print(f"\n✅ VERIFICATION PASSED!")
        print(f"   All class IDs are within valid range [0-{num_classes-1}]")
        return True


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Regenerate masks with semantic segmentation approach"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='outputs/synthetic_data',
        help='Dataset directory'
    )
    parser.add_argument(
        '--start_from',
        type=int,
        default=0,
        help='Sample index to start from (for resuming)'
    )
    parser.add_argument(
        '--verify_only',
        action='store_true',
        help='Only verify existing masks without regenerating'
    )
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    if args.verify_only:
        verify_dataset_integrity(args.data_dir, config)
    else:
        print("\n⚠️  WARNING: This will overwrite all existing masks!")
        response = input("Continue? (yes/no): ")
        
        if response.lower() != 'yes':
            print("Aborted.")
            return
        
        regenerate_all_masks(
            data_dir=args.data_dir,
            config=config,
            start_from=args.start_from,
            verify_only=False
        )
        
        print("\nVerifying regenerated masks...")
        verify_dataset_integrity(args.data_dir, config)


if __name__ == "__main__":
    main()