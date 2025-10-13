"""
Training module for handwritten character segmentation.
Provides Lightning module and callbacks for robust training.
"""

from training.lightning_module import CharacterSegmentationModule
from training.callbacks import (
    CustomModelCheckpoint,
    CustomEarlyStopping,
    MetricLogger,
    GradientLoggingCallback,
    MemoryLoggingCallback,
    SavePredictionsCallback,
    create_callbacks
)

__all__ = [
    'CharacterSegmentationModule',
    'CustomModelCheckpoint',
    'CustomEarlyStopping',
    'MetricLogger',
    'GradientLoggingCallback',
    'MemoryLoggingCallback',
    'SavePredictionsCallback',
    'create_callbacks'
]

__version__ = '1.0.0'