"""
Models module for handwritten character segmentation.
Provides U-Net architecture, loss functions, and evaluation metrics.
"""

from models.unet import (
    UNet,
    ConvBlock,
    AttentionGate,
    Encoder,
    Decoder,
    create_unet,
    count_parameters
)

from models.losses import (
    DiceLoss,
    FocalLoss,
    CombinedLoss,
    TverskyLoss,
    get_loss_function
)

from models.metrics import (
    SegmentationMetrics,
    InstanceSegmentationMetrics,
    compute_iou,
    compute_dice,
    compute_pixel_accuracy,
    create_metrics
)

__all__ = [
    'UNet',
    'ConvBlock',
    'AttentionGate',
    'Encoder',
    'Decoder',
    'create_unet',
    'count_parameters',
    'DiceLoss',
    'FocalLoss',
    'CombinedLoss',
    'TverskyLoss',
    'get_loss_function',
    'SegmentationMetrics',
    'InstanceSegmentationMetrics',
    'compute_iou',
    'compute_dice',
    'compute_pixel_accuracy',
    'create_metrics'
]

__version__ = '1.0.0'