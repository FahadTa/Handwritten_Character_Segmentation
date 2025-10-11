"""
Data module for handwritten character segmentation.
Handles synthetic data generation, text sampling, and mask creation.
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

__all__ = [
    'TextSampler',
    'DatasetGenerator',
    'CharacterAnnotation',
    'ImageMetadata',
    'MaskGenerator',
    'MaskGenerationConfig'
]

__version__ = '1.0.0'