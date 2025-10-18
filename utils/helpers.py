"""
Helper utilities for handwritten character segmentation.
Provides post-processing, refinement, and character extraction functions.
"""

import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional
from scipy import ndimage
from skimage import measure, morphology


class PostProcessor:
    """
    Post-processing pipeline for segmentation masks.
    Refines predictions and extracts individual character instances.
    """
    
    def __init__(
        self,
        min_area: int = 50,
        max_area: int = 50000,
        min_aspect_ratio: float = 0.1,
        max_aspect_ratio: float = 10.0
    ):
        """
        Initialize post-processor.
        
        Args:
            min_area: Minimum character area in pixels
            max_area: Maximum character area in pixels
            min_aspect_ratio: Minimum width/height ratio
            max_aspect_ratio: Maximum width/height ratio
        """
        self.min_area = min_area
        self.max_area = max_area
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
    
    def refine_mask(
        self,
        mask: np.ndarray,
        apply_morphology: bool = True,
        fill_holes: bool = True,
        remove_small: bool = True
    ) -> np.ndarray:
        """
        Refine segmentation mask with morphological operations.
        
        Args:
            mask: Input mask (H, W)
            apply_morphology: Apply opening and closing
            fill_holes: Fill holes in characters
            remove_small: Remove small noise regions
            
        Returns:
            Refined mask
        """
        refined_mask = mask.copy()
        
        if apply_morphology:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel)
            refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel)
        
        if fill_holes:
            refined_mask = self._fill_holes_all_instances(refined_mask)
        
        if remove_small:
            refined_mask = self._remove_small_regions(refined_mask)
        
        return refined_mask
    
    def extract_character_instances(
        self,
        mask: np.ndarray,
        image: Optional[np.ndarray] = None
    ) -> List[Dict]:
        """
        Extract individual character instances from mask.
        
        Args:
            mask: Segmentation mask (H, W)
            image: Optional original image for cropping
            
        Returns:
            List of character dictionaries with bbox, mask, crop, etc.
        """
        unique_ids = np.unique(mask)
        unique_ids = unique_ids[unique_ids != 0]
        
        characters = []
        
        for instance_id in unique_ids:
            instance_mask = (mask == instance_id).astype(np.uint8)
            
            props = measure.regionprops(instance_mask)[0]
            
            area = props.area
            if area < self.min_area or area > self.max_area:
                continue
            
            bbox = props.bbox
            y1, x1, y2, x2 = bbox
            
            width = x2 - x1
            height = y2 - y1
            
            if height == 0 or width == 0:
                continue
            
            aspect_ratio = width / height
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                continue
            
            character_info = {
                'instance_id': int(instance_id),
                'bbox': (x1, y1, x2, y2),
                'center': (props.centroid[1], props.centroid[0]),
                'area': area,
                'width': width,
                'height': height,
                'aspect_ratio': aspect_ratio,
                'mask': instance_mask[y1:y2, x1:x2]
            }
            
            if image is not None:
                character_info['crop'] = image[y1:y2, x1:x2]
            
            characters.append(character_info)
        
        characters = sorted(characters, key=lambda x: (x['center'][1], x['center'][0]))
        
        return characters
    
    def separate_touching_characters(
        self,
        mask: np.ndarray,
        watershed: bool = True
    ) -> np.ndarray:
        """
        Separate touching characters using watershed or distance transform.
        
        Args:
            mask: Binary or instance mask
            watershed: Use watershed algorithm
            
        Returns:
            Separated mask with distinct instances
        """
        if watershed:
            return self._watershed_separation(mask)
        else:
            return self._distance_separation(mask)
    
    def _watershed_separation(self, mask: np.ndarray) -> np.ndarray:
        """
        Separate touching characters using watershed algorithm.
        
        Args:
            mask: Input mask
            
        Returns:
            Separated mask
        """
        binary_mask = (mask > 0).astype(np.uint8)
        
        dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
        
        local_max = morphology.local_maxima(dist_transform)
        markers = measure.label(local_max)
        
        markers = markers + 1
        markers[binary_mask == 0] = 0
        
        labels = morphology.watershed(-dist_transform, markers, mask=binary_mask)
        
        return labels
    
    def _distance_separation(self, mask: np.ndarray) -> np.ndarray:
        """
        Separate using distance transform peaks.
        
        Args:
            mask: Input mask
            
        Returns:
            Separated mask
        """
        binary_mask = (mask > 0).astype(np.uint8)
        
        dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
        
        threshold = dist_transform.max() * 0.4
        _, peaks = cv2.threshold(dist_transform, threshold, 255, cv2.THRESH_BINARY)
        peaks = peaks.astype(np.uint8)
        
        labels = measure.label(peaks)
        
        return labels
    
    def _fill_holes_all_instances(self, mask: np.ndarray) -> np.ndarray:
        """
        Fill holes in all character instances.
        
        Args:
            mask: Instance mask
            
        Returns:
            Mask with filled holes
        """
        filled_mask = mask.copy()
        unique_ids = np.unique(mask)
        
        for instance_id in unique_ids:
            if instance_id == 0:
                continue
            
            instance_mask = (mask == instance_id).astype(bool)
            filled_instance = ndimage.binary_fill_holes(instance_mask)
            filled_mask[filled_instance] = instance_id
        
        return filled_mask
    
    def _remove_small_regions(self, mask: np.ndarray) -> np.ndarray:
        """
        Remove small noise regions from mask.
        
        Args:
            mask: Instance mask
            
        Returns:
            Cleaned mask
        """
        cleaned_mask = np.zeros_like(mask)
        unique_ids = np.unique(mask)
        
        for instance_id in unique_ids:
            if instance_id == 0:
                continue
            
            instance_mask = (mask == instance_id)
            area = instance_mask.sum()
            
            if area >= self.min_area:
                cleaned_mask[instance_mask] = instance_id
        
        return cleaned_mask


def compute_iou_bbox(bbox1: Tuple[int, int, int, int], 
                     bbox2: Tuple[int, int, int, int]) -> float:
    """
    Compute IoU between two bounding boxes.
    
    Args:
        bbox1: First bbox (x1, y1, x2, y2)
        bbox2: Second bbox (x1, y1, x2, y2)
        
    Returns:
        IoU score
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def non_maximum_suppression(
    bboxes: List[Tuple[int, int, int, int]],
    scores: List[float],
    iou_threshold: float = 0.5
) -> List[int]:
    """
    Apply Non-Maximum Suppression to remove overlapping detections.
    
    Args:
        bboxes: List of bounding boxes
        scores: Confidence scores for each bbox
        iou_threshold: IoU threshold for suppression
        
    Returns:
        Indices of boxes to keep
    """
    if len(bboxes) == 0:
        return []
    
    indices = np.argsort(scores)[::-1]
    
    keep = []
    
    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
        
        current_bbox = bboxes[current]
        remaining_indices = indices[1:]
        
        ious = [compute_iou_bbox(current_bbox, bboxes[i]) for i in remaining_indices]
        
        indices = remaining_indices[np.array(ious) < iou_threshold]
    
    return keep


def draw_bboxes_on_image(
    image: np.ndarray,
    bboxes: List[Tuple[int, int, int, int]],
    labels: Optional[List[str]] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw bounding boxes on image.
    
    Args:
        image: Input image (H, W, 3)
        bboxes: List of bounding boxes (x1, y1, x2, y2)
        labels: Optional labels for each bbox
        color: Box color (B, G, R)
        thickness: Line thickness
        
    Returns:
        Image with drawn boxes
    """
    result = image.copy()
    
    for idx, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
        
        if labels and idx < len(labels):
            label = labels[idx]
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            
            text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
            
            cv2.rectangle(
                result,
                (x1, y1 - text_size[1] - 5),
                (x1 + text_size[0], y1),
                color,
                -1
            )
            
            cv2.putText(
                result,
                label,
                (x1, y1 - 5),
                font,
                font_scale,
                (255, 255, 255),
                font_thickness
            )
    
    return result


def extract_line_segments(
    characters: List[Dict],
    line_threshold: int = 50
) -> List[List[Dict]]:
    """
    Group characters into text lines based on vertical position.
    
    Args:
        characters: List of character dictionaries
        line_threshold: Maximum vertical distance for same line
        
    Returns:
        List of lines, each containing character dictionaries
    """
    if not characters:
        return []
    
    sorted_chars = sorted(characters, key=lambda x: x['center'][1])
    
    lines = []
    current_line = [sorted_chars[0]]
    current_y = sorted_chars[0]['center'][1]
    
    for char in sorted_chars[1:]:
        char_y = char['center'][1]
        
        if abs(char_y - current_y) <= line_threshold:
            current_line.append(char)
        else:
            current_line = sorted(current_line, key=lambda x: x['center'][0])
            lines.append(current_line)
            current_line = [char]
            current_y = char_y
    
    if current_line:
        current_line = sorted(current_line, key=lambda x: x['center'][0])
        lines.append(current_line)
    
    return lines


def calculate_character_spacing(characters: List[Dict]) -> Dict[str, float]:
    """
    Calculate spacing statistics for characters.
    
    Args:
        characters: List of character dictionaries
        
    Returns:
        Dictionary with spacing statistics
    """
    if len(characters) < 2:
        return {'mean': 0.0, 'median': 0.0, 'std': 0.0}
    
    sorted_chars = sorted(characters, key=lambda x: x['center'][0])
    
    spacings = []
    for i in range(len(sorted_chars) - 1):
        x1 = sorted_chars[i]['bbox'][2]
        x2 = sorted_chars[i + 1]['bbox'][0]
        spacing = x2 - x1
        if spacing > 0:
            spacings.append(spacing)
    
    if not spacings:
        return {'mean': 0.0, 'median': 0.0, 'std': 0.0}
    
    return {
        'mean': np.mean(spacings),
        'median': np.median(spacings),
        'std': np.std(spacings),
        'spacings': spacings
    }


def save_character_crops(
    characters: List[Dict],
    output_dir: str,
    prefix: str = "char"
):
    """
    Save individual character crops to disk.
    
    Args:
        characters: List of character dictionaries with 'crop' key
        output_dir: Output directory path
        prefix: Filename prefix
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, char in enumerate(characters):
        if 'crop' not in char:
            continue
        
        crop = char['crop']
        filename = f"{prefix}_{idx:04d}.png"
        filepath = os.path.join(output_dir, filename)
        
        cv2.imwrite(filepath, crop)
    
    print(f"Saved {len(characters)} character crops to {output_dir}")


def create_segmentation_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    colormap: str = 'jet'
) -> np.ndarray:
    """
    Create colored overlay of segmentation mask on image.
    
    Args:
        image: Input image (H, W, 3)
        mask: Segmentation mask (H, W)
        alpha: Transparency factor (0-1)
        colormap: Matplotlib colormap name
        
    Returns:
        Overlay image
    """
    import matplotlib.cm as cm
    
    cmap = cm.get_cmap(colormap)
    
    mask_normalized = mask.astype(float) / (mask.max() + 1e-10)
    
    colored_mask = cmap(mask_normalized)[:, :, :3]
    colored_mask = (colored_mask * 255).astype(np.uint8)
    
    overlay = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
    
    return overlay


def resize_with_aspect_ratio(
    image: np.ndarray,
    target_size: Tuple[int, int],
    pad: bool = True,
    pad_value: int = 255
) -> Tuple[np.ndarray, Dict]:
    """
    Resize image while maintaining aspect ratio.
    
    Args:
        image: Input image
        target_size: Target (height, width)
        pad: Whether to pad to exact size
        pad_value: Padding value
        
    Returns:
        Resized image and transformation info
    """
    h, w = image.shape[:2]
    target_h, target_w = target_size
    
    scale = min(target_h / h, target_w / w)
    
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    if pad:
        if len(image.shape) == 3:
            padded = np.full((target_h, target_w, image.shape[2]), pad_value, dtype=image.dtype)
        else:
            padded = np.full((target_h, target_w), pad_value, dtype=image.dtype)
        
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        transform_info = {
            'scale': scale,
            'offset': (x_offset, y_offset),
            'original_size': (h, w),
            'resized_size': (new_h, new_w)
        }
        
        return padded, transform_info
    else:
        transform_info = {
            'scale': scale,
            'original_size': (h, w),
            'resized_size': (new_h, new_w)
        }
        return resized, transform_info