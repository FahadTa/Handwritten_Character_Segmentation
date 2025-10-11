import os
import json
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, asdict

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from data.text_sampler import TextSampler


@dataclass
class CharacterAnnotation:
    """Annotation for a single character in the image."""
    character: str
    char_index: int
    class_id: int
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    center: Tuple[float, float]  # (cx, cy)
    font_size: int
    line_number: int


@dataclass
class ImageMetadata:
    """Metadata for a generated synthetic image."""
    image_id: str
    image_path: str
    mask_path: str
    width: int
    height: int
    text_content: str
    num_characters: int
    font_name: str
    font_size: int
    annotations: List[Dict]


class DatasetGenerator:
    """
    Synthetic handwritten dataset generator.
    Renders text with handwritten fonts and tracks character positions.
    """
    
    def __init__(
        self,
        fonts_dir: str,
        output_dir: str,
        image_size: Tuple[int, int],
        character_set: str,
        background_color: Tuple[int, int, int] = (255, 255, 255),
        text_color_range: List[Tuple[int, int, int]] = None,
        margin: Tuple[int, int] = (50, 50),
        seed: int = 42
    ):
        """
        Initialize dataset generator.
        
        Args:
            fonts_dir: Directory containing .ttf font files
            output_dir: Output directory for generated data
            image_size: (height, width) of output images
            character_set: String of all allowed characters
            background_color: RGB background color
            text_color_range: Range of text colors [(min_r, min_g, min_b), (max_r, max_g, max_b)]
            margin: (vertical, horizontal) margins in pixels
            seed: Random seed
        """
        self.fonts_dir = Path(fonts_dir)
        self.output_dir = Path(output_dir)
        self.image_size = image_size
        self.character_set = character_set
        self.background_color = background_color
        self.text_color_range = text_color_range or [[(0, 0, 0), (50, 50, 50)]]
        self.margin = margin
        self.seed = seed
        
        random.seed(seed)
        np.random.seed(seed)
        
        self.char_to_id = {char: idx for idx, char in enumerate(character_set)}
        self.id_to_char = {idx: char for char, idx in self.char_to_id.items()}
        
        self.fonts = self._load_fonts()
        
        self._create_output_directories()
    
    def _load_fonts(self) -> List[Path]:
        """
        Load all .ttf font files from fonts directory.
        
        Returns:
            List of font file paths
        """
        if not self.fonts_dir.exists():
            raise FileNotFoundError(f"Fonts directory not found: {self.fonts_dir}")
        
        font_files = list(self.fonts_dir.glob("*.ttf"))
        font_files.extend(list(self.fonts_dir.glob("*.TTF")))
        
        if not font_files:
            raise ValueError(f"No .ttf font files found in {self.fonts_dir}")
        
        print(f"Loaded {len(font_files)} font files")
        return font_files
    
    def _create_output_directories(self) -> None:
        """Create output directory structure."""
        (self.output_dir / "images").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "masks").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "metadata").mkdir(parents=True, exist_ok=True)
    
    def _sample_text_color(self) -> Tuple[int, int, int]:
        """
        Sample random text color from configured range.
        
        Returns:
            RGB color tuple
        """
        min_color, max_color = self.text_color_range[0]
        
        r = random.randint(min_color[0], max_color[0])
        g = random.randint(min_color[1], max_color[1])
        b = random.randint(min_color[2], max_color[2])
        
        return (r, g, b)
    
    def _get_char_bbox(
        self,
        draw: ImageDraw.Draw,
        char: str,
        position: Tuple[int, int],
        font: ImageFont.FreeTypeFont
    ) -> Tuple[int, int, int, int]:
        """
        Get precise bounding box for a character.
        
        Args:
            draw: ImageDraw object
            char: Character to measure
            position: (x, y) position where character is drawn
            font: Font object
            
        Returns:
            (x1, y1, x2, y2) bounding box
        """
        x, y = position
        
        bbox = draw.textbbox((x, y), char, font=font)
        
        return bbox
    
    def _render_text_line(
        self,
        draw: ImageDraw.Draw,
        text: str,
        position: Tuple[int, int],
        font: ImageFont.FreeTypeFont,
        color: Tuple[int, int, int],
        char_spacing: int,
        line_number: int,
        annotations: List[CharacterAnnotation]
    ) -> Tuple[int, int]:
        """
        Render a single line of text and track character positions.
        
        Args:
            draw: ImageDraw object
            text: Text to render
            position: Starting (x, y) position
            font: Font object
            color: Text color
            char_spacing: Additional spacing between characters
            line_number: Line number for annotation
            annotations: List to append character annotations
            
        Returns:
            (width, height) of rendered text
        """
        x, y = position
        start_x = x
        max_height = 0
        
        char_index = len(annotations)
        
        for char in text:
            if char not in self.char_to_id:
                continue
            
            draw.text((x, y), char, font=font, fill=color)
            
            bbox = self._get_char_bbox(draw, char, (x, y), font)
            x1, y1, x2, y2 = bbox
            
            char_width = x2 - x1
            char_height = y2 - y1
            max_height = max(max_height, char_height)
            
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            annotation = CharacterAnnotation(
                character=char,
                char_index=char_index,
                class_id=self.char_to_id[char],
                bbox=(x1, y1, x2, y2),
                center=(center_x, center_y),
                font_size=font.size,
                line_number=line_number
            )
            annotations.append(annotation)
            
            x += char_width + char_spacing
            char_index += 1
        
        total_width = x - start_x
        
        return total_width, max_height
    
    def _split_text_into_lines(
        self,
        text: str,
        font: ImageFont.FreeTypeFont,
        max_width: int,
        draw: ImageDraw.Draw
    ) -> List[str]:
        """
        Split text into lines that fit within max_width.
        
        Args:
            text: Text to split
            font: Font object
            max_width: Maximum line width in pixels
            draw: ImageDraw object for measurement
            
        Returns:
            List of text lines
        """
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            bbox = draw.textbbox((0, 0), test_line, font=font)
            width = bbox[2] - bbox[0]
            
            if width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    lines.append(word)
                    current_line = []
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines
    
    def generate_image(
        self,
        text: str,
        font_size: int,
        line_spacing: float,
        char_spacing: int,
        font_path: Optional[Path] = None
    ) -> Tuple[Image.Image, List[CharacterAnnotation]]:
        """
        Generate single synthetic image with text.
        
        Args:
            text: Text to render
            font_size: Font size in pixels
            line_spacing: Line spacing multiplier
            char_spacing: Additional character spacing
            font_path: Specific font to use (random if None)
            
        Returns:
            (image, annotations) tuple
        """
        height, width = self.image_size
        
        image = Image.new('RGB', (width, height), color=self.background_color)
        draw = ImageDraw.Draw(image)
        
        if font_path is None:
            font_path = random.choice(self.fonts)
        
        try:
            font = ImageFont.truetype(str(font_path), font_size)
        except Exception as e:
            print(f"Warning: Failed to load font {font_path}: {e}")
            font = ImageFont.load_default()
        
        text_color = self._sample_text_color()
        
        margin_v, margin_h = self.margin
        max_text_width = width - 2 * margin_h
        
        lines = self._split_text_into_lines(text, font, max_text_width, draw)
        
        annotations = []
        
        y_position = margin_v
        
        for line_idx, line in enumerate(lines):
            if y_position >= height - margin_v:
                break
            
            x_position = margin_h
            
            line_width, line_height = self._render_text_line(
                draw=draw,
                text=line,
                position=(x_position, y_position),
                font=font,
                color=text_color,
                char_spacing=char_spacing,
                line_number=line_idx,
                annotations=annotations
            )
            
            y_position += int(line_height * line_spacing)
        
        return image, annotations
    
    def generate_dataset(
        self,
        text_sampler: TextSampler,
        num_samples: int,
        min_chars: int,
        max_chars: int,
        font_size_range: Tuple[int, int],
        line_spacing_range: Tuple[float, float],
        char_spacing_range: Tuple[int, int]
    ) -> List[ImageMetadata]:
        """
        Generate complete synthetic dataset.
        
        Args:
            text_sampler: TextSampler instance for text generation
            num_samples: Number of images to generate
            min_chars: Minimum characters per image
            max_chars: Maximum characters per image
            font_size_range: (min, max) font size
            line_spacing_range: (min, max) line spacing multiplier
            char_spacing_range: (min, max) character spacing
            
        Returns:
            List of image metadata
        """
        metadata_list = []
        
        print(f"Generating {num_samples} synthetic images...")
        
        for sample_idx in tqdm(range(num_samples), desc="Generating images"):
            num_chars = random.randint(min_chars, max_chars)
            text = text_sampler.sample_text(num_chars=num_chars)
            
            font_size = random.randint(*font_size_range)
            line_spacing = random.uniform(*line_spacing_range)
            char_spacing = random.randint(*char_spacing_range)
            font_path = random.choice(self.fonts)
            
            image, annotations = self.generate_image(
                text=text,
                font_size=font_size,
                line_spacing=line_spacing,
                char_spacing=char_spacing,
                font_path=font_path
            )
            
            image_id = f"sample_{sample_idx:06d}"
            image_filename = f"{image_id}.png"
            mask_filename = f"{image_id}_mask.png"
            
            image_path = self.output_dir / "images" / image_filename
            mask_path = self.output_dir / "masks" / mask_filename
            
            image.save(image_path)
            
            metadata = ImageMetadata(
                image_id=image_id,
                image_path=str(image_path),
                mask_path=str(mask_path),
                width=self.image_size[1],
                height=self.image_size[0],
                text_content=text,
                num_characters=len(annotations),
                font_name=font_path.stem,
                font_size=font_size,
                annotations=[asdict(ann) for ann in annotations]
            )
            
            metadata_path = self.output_dir / "metadata" / f"{image_id}.json"
            with open(metadata_path, 'w') as f:
                json.dump(asdict(metadata), f, indent=2)
            
            metadata_list.append(metadata)
        
        self._save_dataset_summary(metadata_list)
        
        return metadata_list
    
    def _save_dataset_summary(self, metadata_list: List[ImageMetadata]) -> None:
        """
        Save dataset summary statistics.
        
        Args:
            metadata_list: List of image metadata
        """
        summary = {
            "total_images": len(metadata_list),
            "total_characters": sum(m.num_characters for m in metadata_list),
            "character_set": self.character_set,
            "num_classes": len(self.character_set),
            "char_to_id": self.char_to_id,
            "image_size": self.image_size,
            "fonts_used": list(set(m.font_name for m in metadata_list))
        }
        
        summary_path = self.output_dir / "dataset_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nDataset generation complete!")
        print(f"Total images: {summary['total_images']}")
        print(f"Total characters: {summary['total_characters']}")
        print(f"Summary saved to: {summary_path}")