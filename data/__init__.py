"""
Data module for handwritten character segmentation.
Handles synthetic data generation, text sampling, mask creation,
augmentation, and data loading.
"""

from data.text_sampler import TextSampler
from data.dataset_generator import (
    DatasetGenerator,
    CharacterAnnotation,
    ImageMetadata
)
from data.mask_generator import (
    MaskGenerator,
    MaskGenerationConfig
)
from data.augmentations import (
    SegmentationAugmentation,
    DocumentAugmentation,
    get_augmentation_pipeline
)
from data.dataset import (
    CharacterSegmentationDataset,
    CharacterSegmentationInferenceDataset,
    collate_fn,
    get_dataset
)
from data.datamodule import (
    CharacterSegmentationDataModule,
    InferenceDataModule,
    create_datamodule
)

__all__ = [
    'TextSampler',
    'DatasetGenerator',
    'CharacterAnnotation',
    'ImageMetadata',
    'MaskGenerator',
    'MaskGenerationConfig',
    'SegmentationAugmentation',
    'DocumentAugmentation',
    'get_augmentation_pipeline',
    'CharacterSegmentationDataset',
    'CharacterSegmentationInferenceDataset',
    'collate_fn',
    'get_dataset',
    'CharacterSegmentationDataModule',
    'InferenceDataModule',
    'create_datamodule'
]

__version__ = '1.0.0'