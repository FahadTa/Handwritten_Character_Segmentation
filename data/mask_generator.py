import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage


@dataclass
class MaskGenerationConfig:
    """Configuration for mask generation."""
    method: str = "template_matching"
    template_threshold: float = 0.7
    dilation_kernel_size: int = 3
    use_morphology: bool = True
    fill_holes: bool = True
    min_char_area: int = 10


class MaskGenerator:
    """
    Generates pixel-perfect segmentation masks for character-level instance segmentation.
    Each character occurrence gets a unique instance ID.
    """
    
    def __init__(
        self,
        character_set: str,
        image_size: Tuple[int, int],
        config: Optional[MaskGenerationConfig] = None
    ):
        """
        Initialize mask generator.
        
        Args:
            character_set: String of all valid characters
            image_size: (height, width) of images
            config: Mask generation configuration
        """
        self.character_set = character_set
        self.image_size = image_size
        self.config = config or MaskGenerationConfig()
        
        self.char_to_id = {char: idx + 1 for idx, char in enumerate(character_set)}
        self.char_to_id[' '] = 0
        
        self.num_classes = len(self.char_to_id)
    
    def generate_mask_from_annotations(
        self,
        image: np.ndarray,
        annotations: List[Dict],
        font_path: str,
        font_size: int
    ) -> np.ndarray:
        """
        Generate instance segmentation mask from character annotations.
        Each character gets a unique instance ID based on its position.
        
        Args:
            image: Original image as numpy array (H, W, 3)
            annotations: List of character annotation dictionaries
            font_path: Path to font file used for rendering
            font_size: Font size used in image
            
        Returns:
            Instance segmentation mask (H, W) with unique IDs per character
        """
        if self.config.method == "template_matching":
            return self._generate_mask_template_matching(
                image, annotations, font_path, font_size
            )
        elif self.config.method == "bounding_box":
            return self._generate_mask_bounding_box(image, annotations)
        else:
            raise ValueError(f"Unknown mask generation method: {self.config.method}")
    
    def _generate_mask_bounding_box(
        self,
        image: np.ndarray,
        annotations: List[Dict]
    ) -> np.ndarray:
        """
        Generate instance mask using bounding box method.
        Each character gets a unique instance ID.
        
        Args:
            image: Original image
            annotations: Character annotations
            
        Returns:
            Instance segmentation mask
        """
        height, width = self.image_size
        mask = np.zeros((height, width), dtype=np.int32)
        
        for ann in annotations:
            char = ann['character']
            
            if char == ' ':
                continue
            
            instance_id = ann['char_index'] + 1
            
            x1, y1, x2, y2 = ann['bbox']
            
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(width, int(x2))
            y2 = min(height, int(y2))
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            mask[y1:y2, x1:x2] = instance_id
        
        if self.config.use_morphology:
            mask = self._apply_morphology(mask)
        
        return mask
    
    def _generate_mask_template_matching(
        self,
        image: np.ndarray,
        annotations: List[Dict],
        font_path: str,
        font_size: int
    ) -> np.ndarray:
        """
        Generate instance mask using template matching for pixel-perfect accuracy.
        Each character gets a unique instance ID.
        
        Args:
            image: Original image (H, W, 3)
            annotations: Character annotations
            font_path: Path to font file
            font_size: Font size
            
        Returns:
            Instance segmentation mask (H, W)
        """
        height, width = self.image_size
        mask = np.zeros((height, width), dtype=np.int32)
        
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception as e:
            print(f"Warning: Failed to load font {font_path}: {e}")
            return self._generate_mask_bounding_box(image, annotations)
        
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        for ann in annotations:
            char = ann['character']
            
            if char == ' ':
                continue
            
            instance_id = ann['char_index'] + 1
            
            x1, y1, x2, y2 = ann['bbox']
            
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(width, int(x2))
            y2 = min(height, int(y2))
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            char_region = image_gray[y1:y2, x1:x2]
            
            if char_region.size == 0:
                continue
            
            template = self._create_character_template(
                char, font, (x2 - x1, y2 - y1)
            )
            
            if template is None or template.shape[0] == 0 or template.shape[1] == 0:
                mask[y1:y2, x1:x2] = instance_id
                continue
            
            char_mask = self._match_template_to_region(
                char_region, template, self.config.template_threshold
            )
            
            if char_mask is not None:
                mask[y1:y2, x1:x2] = np.where(char_mask > 0, instance_id, mask[y1:y2, x1:x2])
            else:
                mask[y1:y2, x1:x2] = instance_id
        
        if self.config.use_morphology:
            mask = self._apply_morphology(mask)
        
        if self.config.fill_holes:
            mask = self._fill_holes(mask)
        
        return mask
    
    def _create_character_template(
        self,
        char: str,
        font: ImageFont.FreeTypeFont,
        target_size: Tuple[int, int]
    ) -> Optional[np.ndarray]:
        """
        Create a template image of a single character.
        
        Args:
            char: Character to render
            font: Font object
            target_size: (width, height) of template
            
        Returns:
            Template as grayscale numpy array or None if failed
        """
        try:
            width, height = target_size
            
            padding = 10
            template_img = Image.new(
                'L', 
                (width + 2 * padding, height + 2 * padding), 
                color=255
            )
            draw = ImageDraw.Draw(template_img)
            
            bbox = draw.textbbox((0, 0), char, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            x = padding + (width - text_width) // 2
            y = padding + (height - text_height) // 2
            
            draw.text((x, y), char, font=font, fill=0)
            
            template_array = np.array(template_img)
            
            template_array = 255 - template_array
            
            y_coords, x_coords = np.where(template_array > 0)
            if len(y_coords) == 0:
                return None
            
            y_min, y_max = y_coords.min(), y_coords.max() + 1
            x_min, x_max = x_coords.min(), x_coords.max() + 1
            template_array = template_array[y_min:y_max, x_min:x_max]
            
            return template_array
            
        except Exception as e:
            print(f"Warning: Failed to create template for '{char}': {e}")
            return None
    
    def _match_template_to_region(
        self,
        region: np.ndarray,
        template: np.ndarray,
        threshold: float
    ) -> Optional[np.ndarray]:
        """
        Match template to region using normalized cross-correlation.
        
        Args:
            region: Image region to match against
            template: Template to match
            threshold: Matching threshold (0-1)
            
        Returns:
            Binary mask of matched region or None if matching failed
        """
        try:
            if region.shape[0] < template.shape[0] or region.shape[1] < template.shape[1]:
                template = cv2.resize(
                    template, 
                    (min(region.shape[1], template.shape[1]),
                     min(region.shape[0], template.shape[0])),
                    interpolation=cv2.INTER_AREA
                )
            
            region_binary = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            template_binary = (template > 0).astype(np.uint8) * 255
            
            if template_binary.shape[0] > region_binary.shape[0] or \
               template_binary.shape[1] > region_binary.shape[1]:
                return (region_binary > 0).astype(np.uint8)
            
            result = cv2.matchTemplate(
                region_binary, 
                template_binary, 
                cv2.TM_CCOEFF_NORMED
            )
            
            max_val = result.max()
            
            if max_val < threshold:
                return (region_binary > 0).astype(np.uint8)
            
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            top_left = max_loc
            h, w = template_binary.shape
            
            char_mask = np.zeros_like(region_binary, dtype=np.uint8)
            
            y1, y2 = top_left[1], top_left[1] + h
            x1, x2 = top_left[0], top_left[0] + w
            
            char_mask[y1:y2, x1:x2] = template_binary
            
            return (char_mask > 0).astype(np.uint8)
            
        except Exception as e:
            print(f"Warning: Template matching failed: {e}")
            return None
    
    def _apply_morphology(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations to clean up instance mask.
        
        Args:
            mask: Input instance mask
            
        Returns:
            Cleaned mask
        """
        kernel_size = self.config.dilation_kernel_size
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (kernel_size, kernel_size)
        )
        
        unique_instances = np.unique(mask)
        result_mask = np.zeros_like(mask)
        
        for instance_id in unique_instances:
            if instance_id == 0:
                continue
            
            instance_mask = (mask == instance_id).astype(np.uint8)
            
            instance_mask = cv2.morphologyEx(
                instance_mask, 
                cv2.MORPH_CLOSE, 
                kernel
            )
            
            result_mask[instance_mask > 0] = instance_id
        
        return result_mask
    
    def _fill_holes(self, mask: np.ndarray) -> np.ndarray:
        """
        Fill holes in character instance masks.
        
        Args:
            mask: Input instance mask
            
        Returns:
            Mask with filled holes
        """
        unique_instances = np.unique(mask)
        result_mask = mask.copy()
        
        for instance_id in unique_instances:
            if instance_id == 0:
                continue
            
            instance_mask = (mask == instance_id).astype(bool)
            
            filled_mask = ndimage.binary_fill_holes(instance_mask)
            
            result_mask[filled_mask] = instance_id
        
        return result_mask
    
    def visualize_mask(
        self,
        mask: np.ndarray,
        num_colors: Optional[int] = None
    ) -> np.ndarray:
        """
        Create RGB visualization of instance segmentation mask.
        Each instance gets a unique random color.
        
        Args:
            mask: Instance segmentation mask (H, W)
            num_colors: Number of colors to use (None = auto)
            
        Returns:
            RGB visualization (H, W, 3)
        """
        max_instance = mask.max()
        
        if num_colors is None:
            num_colors = max_instance + 1
        
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(num_colors, 3), dtype=np.uint8)
        colors[0] = [0, 0, 0]
        
        rgb_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        
        for instance_id in range(num_colors):
            if instance_id < len(colors):
                rgb_mask[mask == instance_id] = colors[instance_id]
        
        return rgb_mask
    
    def save_mask(self, mask: np.ndarray, output_path: str) -> None:
        """
        Save instance segmentation mask to file.
        
        Args:
            mask: Instance segmentation mask
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        mask_img = Image.fromarray(mask.astype(np.int32), mode='I')
        mask_img.save(output_path)
    
    def load_mask(self, mask_path: str) -> np.ndarray:
        """
        Load instance segmentation mask from file.
        
        Args:
            mask_path: Path to mask file
            
        Returns:
            Instance segmentation mask
        """
        mask_img = Image.open(mask_path)
        mask = np.array(mask_img, dtype=np.int32)
        
        return mask
    
    def get_mask_statistics(self, mask: np.ndarray) -> Dict:
        """
        Compute statistics about instance segmentation mask.
        
        Args:
            mask: Instance segmentation mask
            
        Returns:
            Dictionary with statistics
        """
        unique_instances = np.unique(mask)
        
        num_instances = len(unique_instances) - 1 if 0 in unique_instances else len(unique_instances)
        
        total_pixels = mask.size
        character_pixels = np.sum(mask > 0)
        background_pixels = np.sum(mask == 0)
        
        instance_sizes = {}
        for instance_id in unique_instances:
            if instance_id == 0:
                continue
            instance_sizes[int(instance_id)] = int(np.sum(mask == instance_id))
        
        return {
            "num_characters": num_instances,
            "total_pixels": total_pixels,
            "character_pixels": int(character_pixels),
            "background_pixels": int(background_pixels),
            "character_ratio": float(character_pixels / total_pixels),
            "instance_sizes": instance_sizes
        }