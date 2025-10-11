"""
Production script for generating synthetic handwritten character dataset.
Generates images, masks, and metadata with proper train/val/test splits.
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List
from datetime import datetime

import numpy as np
from PIL import Image
from tqdm import tqdm

from config import load_config
from data import (
    TextSampler,
    DatasetGenerator,
    MaskGenerator,
    MaskGenerationConfig
)


def setup_logging(log_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_dir: Directory for log files
        log_level: Logging level
        
    Returns:
        Configured logger
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"dataset_generation_{timestamp}.log"
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_file}")
    
    return logger


def split_dataset(
    metadata_list: List,
    train_split: float,
    val_split: float,
    test_split: float,
    seed: int
) -> Dict[str, List]:
    """
    Split dataset into train/val/test sets.
    
    Args:
        metadata_list: List of image metadata
        train_split: Training set ratio
        val_split: Validation set ratio
        test_split: Test set ratio
        seed: Random seed
        
    Returns:
        Dictionary with 'train', 'val', 'test' keys
    """
    np.random.seed(seed)
    
    indices = np.random.permutation(len(metadata_list))
    
    train_end = int(len(indices) * train_split)
    val_end = train_end + int(len(indices) * val_split)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    splits = {
        'train': [metadata_list[i] for i in train_indices],
        'val': [metadata_list[i] for i in val_indices],
        'test': [metadata_list[i] for i in test_indices]
    }
    
    return splits


def save_split_info(splits: Dict[str, List], output_dir: Path, logger: logging.Logger):
    """
    Save dataset split information.
    
    Args:
        splits: Dictionary with train/val/test splits
        output_dir: Output directory
        logger: Logger instance
    """
    split_info = {
        'train': {
            'num_samples': len(splits['train']),
            'image_ids': [m.image_id for m in splits['train']]
        },
        'val': {
            'num_samples': len(splits['val']),
            'image_ids': [m.image_id for m in splits['val']]
        },
        'test': {
            'num_samples': len(splits['test']),
            'image_ids': [m.image_id for m in splits['test']]
        }
    }
    
    split_file = output_dir / "dataset_splits.json"
    with open(split_file, 'w') as f:
        json.dump(split_info, f, indent=2)
    
    logger.info(f"Dataset splits saved to {split_file}")
    logger.info(f"Train: {split_info['train']['num_samples']} samples")
    logger.info(f"Val: {split_info['val']['num_samples']} samples")
    logger.info(f"Test: {split_info['test']['num_samples']} samples")


def generate_masks_for_dataset(
    metadata_list: List,
    mask_generator: MaskGenerator,
    logger: logging.Logger
):
    """
    Generate segmentation masks for all images.
    
    Args:
        metadata_list: List of image metadata
        mask_generator: MaskGenerator instance
        logger: Logger instance
    """
    logger.info("Generating segmentation masks...")
    
    for metadata in tqdm(metadata_list, desc="Generating masks"):
        try:
            image = Image.open(metadata.image_path)
            image_array = np.array(image)
            
            mask = mask_generator.generate_mask_from_annotations(
                image=image_array,
                annotations=metadata.annotations,
                font_path=str(Path("fonts") / f"{metadata.font_name}.ttf"),
                font_size=metadata.font_size
            )
            
            mask_generator.save_mask(mask, metadata.mask_path)
            
            mask_stats = mask_generator.get_mask_statistics(mask)
            
            if mask_stats['num_characters'] != metadata.num_characters:
                logger.warning(
                    f"Character count mismatch in {metadata.image_id}: "
                    f"expected {metadata.num_characters}, got {mask_stats['num_characters']}"
                )
            
        except Exception as e:
            logger.error(f"Failed to generate mask for {metadata.image_id}: {e}")
            continue
    
    logger.info("Mask generation complete!")


def validate_dataset(
    metadata_list: List,
    logger: logging.Logger
) -> Dict:
    """
    Validate generated dataset and compute statistics.
    
    Args:
        metadata_list: List of image metadata
        logger: Logger instance
        
    Returns:
        Dictionary with validation statistics
    """
    logger.info("Validating dataset...")
    
    total_images = len(metadata_list)
    total_characters = sum(m.num_characters for m in metadata_list)
    
    missing_images = []
    missing_masks = []
    
    for metadata in metadata_list:
        if not Path(metadata.image_path).exists():
            missing_images.append(metadata.image_id)
        if not Path(metadata.mask_path).exists():
            missing_masks.append(metadata.image_id)
    
    char_distribution = {}
    for metadata in metadata_list:
        for ann in metadata.annotations:
            char = ann['character']
            char_distribution[char] = char_distribution.get(char, 0) + 1
    
    validation_stats = {
        'total_images': total_images,
        'total_characters': total_characters,
        'missing_images': len(missing_images),
        'missing_masks': len(missing_masks),
        'character_distribution': char_distribution,
        'avg_chars_per_image': total_characters / total_images if total_images > 0 else 0
    }
    
    logger.info(f"Validation complete:")
    logger.info(f"  Total images: {total_images}")
    logger.info(f"  Total characters: {total_characters}")
    logger.info(f"  Avg characters per image: {validation_stats['avg_chars_per_image']:.2f}")
    logger.info(f"  Missing images: {len(missing_images)}")
    logger.info(f"  Missing masks: {len(missing_masks)}")
    
    if missing_images:
        logger.warning(f"Missing image files: {missing_images}")
    if missing_masks:
        logger.warning(f"Missing mask files: {missing_masks}")
    
    return validation_stats


def save_final_report(
    output_dir: Path,
    config_dict: Dict,
    validation_stats: Dict,
    logger: logging.Logger
):
    """
    Save final generation report.
    
    Args:
        output_dir: Output directory
        config_dict: Configuration dictionary
        validation_stats: Validation statistics
        logger: Logger instance
    """
    report = {
        'generation_timestamp': datetime.now().isoformat(),
        'configuration': config_dict,
        'validation_statistics': validation_stats
    }
    
    report_file = output_dir / "generation_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Generation report saved to {report_file}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic handwritten character dataset"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=None,
        help='Number of samples to generate (overrides config)'
    )
    parser.add_argument(
        '--skip_masks',
        action='store_true',
        help='Skip mask generation (only generate images)'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Save mask visualizations for inspection'
    )
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    logger = setup_logging(
        Path(config.get('paths.logs_dir')),
        config.get('logging.local.log_level')
    )
    
    logger.info("=" * 80)
    logger.info("HANDWRITTEN CHARACTER SEGMENTATION - DATASET GENERATION")
    logger.info("=" * 80)
    
    logger.info("Loading configuration...")
    config.print_config()
    
    num_samples = args.num_samples or config.get('dataset_generation.num_samples')
    
    logger.info(f"Generating {num_samples} samples...")
    
    try:
        logger.info("Step 1: Initializing text sampler...")
        text_sampler = TextSampler(
            source=config.get('text_corpus.source'),
            character_set=config.get('dataset_generation.character_set'),
            min_sentence_length=config.get('text_corpus.min_sentence_length'),
            max_sentence_length=config.get('text_corpus.max_sentence_length'),
            language=config.get('text_corpus.wikipedia_language'),
            seed=config.get('project.seed')
        )
        
        logger.info("Step 2: Building text corpus...")
        if config.get('text_corpus.source') in ['wikipedia', 'mixed']:
            text_sampler.build_corpus(
                num_wikipedia_articles=config.get('text_corpus.wikipedia_articles'),
                custom_file_path=config.get('text_corpus.custom_text_file')
            )
        else:
            text_sampler.build_corpus(
                custom_file_path=config.get('text_corpus.custom_text_file')
            )
        
        corpus_stats = text_sampler.get_corpus_stats()
        logger.info(f"Corpus statistics: {corpus_stats}")
        
        logger.info("Step 3: Initializing dataset generator...")
        dataset_generator = DatasetGenerator(
            fonts_dir=config.get('paths.fonts_dir'),
            output_dir=config.get('paths.synthetic_data_dir'),
            image_size=tuple(config.get('dataset_generation.image_size')),
            character_set=config.get('dataset_generation.character_set'),
            background_color=tuple(config.get('dataset_generation.background_color')),
            text_color_range=[tuple(config.get('dataset_generation.text_color_range'))],
            margin=tuple(config.get('dataset_generation.margin')),
            seed=config.get('project.seed')
        )
        
        logger.info("Step 4: Generating synthetic images...")
        metadata_list = dataset_generator.generate_dataset(
            text_sampler=text_sampler,
            num_samples=num_samples,
            min_chars=config.get('dataset_generation.min_chars_per_image'),
            max_chars=config.get('dataset_generation.max_chars_per_image'),
            font_size_range=tuple(config.get('dataset_generation.font_size_range')),
            line_spacing_range=tuple(config.get('dataset_generation.line_spacing_range')),
            char_spacing_range=tuple(config.get('dataset_generation.char_spacing_range'))
        )
        
        if not args.skip_masks:
            logger.info("Step 5: Initializing mask generator...")
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
            
            logger.info("Step 6: Generating segmentation masks...")
            generate_masks_for_dataset(metadata_list, mask_generator, logger)
            
            if args.visualize:
                logger.info("Step 7: Generating mask visualizations...")
                vis_dir = Path(config.get('paths.output_dir')) / "visualizations"
                vis_dir.mkdir(parents=True, exist_ok=True)
                
                sample_size = min(10, len(metadata_list))
                for metadata in tqdm(metadata_list[:sample_size], desc="Creating visualizations"):
                    mask = mask_generator.load_mask(metadata.mask_path)
                    vis_mask = mask_generator.visualize_mask(mask)
                    
                    vis_path = vis_dir / f"{metadata.image_id}_vis.png"
                    Image.fromarray(vis_mask).save(vis_path)
                
                logger.info(f"Visualizations saved to {vis_dir}")
        
        logger.info("Step 8: Creating dataset splits...")
        splits = split_dataset(
            metadata_list,
            config.get('dataset_generation.train_split'),
            config.get('dataset_generation.val_split'),
            config.get('dataset_generation.test_split'),
            config.get('project.seed')
        )
        
        save_split_info(
            splits,
            Path(config.get('paths.synthetic_data_dir')),
            logger
        )
        
        logger.info("Step 9: Validating dataset...")
        validation_stats = validate_dataset(metadata_list, logger)
        
        logger.info("Step 10: Saving final report...")
        save_final_report(
            Path(config.get('paths.synthetic_data_dir')),
            config.to_dict(),
            validation_stats,
            logger
        )
        
        logger.info("=" * 80)
        logger.info("DATASET GENERATION COMPLETE!")
        logger.info("=" * 80)
        logger.info(f"Output directory: {config.get('paths.synthetic_data_dir')}")
        logger.info(f"Total samples: {len(metadata_list)}")
        logger.info(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
        
    except Exception as e:
        logger.error(f"Dataset generation failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()