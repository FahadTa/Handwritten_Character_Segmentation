"""
Utility modules for handwritten character segmentation.
Provides visualization, post-processing, and helper functions.
"""

from utils.visualization import (
    SegmentationVisualizer,
    save_predictions_grid
)

from utils.helpers import (
    PostProcessor,
    compute_iou_bbox,
    non_maximum_suppression,
    draw_bboxes_on_image,
    extract_line_segments,
    calculate_character_spacing,
    save_character_crops,
    create_segmentation_overlay,
    resize_with_aspect_ratio
)

__all__ = [
    'SegmentationVisualizer',
    'save_predictions_grid',
    'PostProcessor',
    'compute_iou_bbox',
    'non_maximum_suppression',
    'draw_bboxes_on_image',
    'extract_line_segments',
    'calculate_character_spacing',
    'save_character_crops',
    'create_segmentation_overlay',
    'resize_with_aspect_ratio'
]

__version__ = '1.0.0'