"""
Inference script for handwritten character segmentation.
Loads trained model and performs segmentation on new images.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import torch
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

from config import load_config
from training import CharacterSegmentationModule
from data.augmentations import get_augmentation_pipeline
from utils import (
    SegmentationVisualizer,
    PostProcessor,
    draw_bboxes_on_image,
    extract_line_segments,
    calculate_character_spacing,
    save_character_crops,
    resize_with_aspect_ratio
)


class CharacterSegmentationInference:
    """
    Inference pipeline for character segmentation.
    Handles model loading, prediction, and post-processing.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        config_path: str = 'config/config.yaml',
        device: Optional[str] = None
    ):
        """
        Initialize inference pipeline.
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            config_path: Path to configuration file
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
        """
        print("=" * 80)
        print("HANDWRITTEN CHARACTER SEGMENTATION - INFERENCE")
        print("=" * 80)
        
        self.config = load_config(config_path)
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"\nLoading model from: {checkpoint_path}")
        self.model = self._load_model(checkpoint_path)
        print(f"Model loaded successfully on {self.device}")
        
        self.transform = get_augmentation_pipeline(
            self.config.to_dict(),
            training=False
        )
        
        self.post_processor = PostProcessor(
            min_area=self.config.get('inference.min_char_size', 10),
            max_area=50000
        )
        
        self.visualizer = SegmentationVisualizer()
        
        self.image_size = tuple(
            self.config.get('dataset_generation.image_size', [512, 512])
        )
        
        print(f"Inference pipeline initialized")
        print(f"Target image size: {self.image_size}")
        print("=" * 80 + "\n")
    
    def _load_model(self, checkpoint_path: str) -> torch.nn.Module:
        """
        Load trained model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Loaded model in eval mode
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        model = CharacterSegmentationModule.load_from_checkpoint(
            str(checkpoint_path),
            config=self.config.to_dict()
        )
        
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def preprocess_image(
        self,
        image_path: str,
        resize: bool = True
    ) -> Dict[str, Any]:
        """
        Load and preprocess image for inference.
        
        Args:
            image_path: Path to input image
            resize: Whether to resize to model input size
            
        Returns:
            Dictionary with image tensor and metadata
        """
        image = Image.open(image_path)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        original_size = image.size
        image_array = np.array(image, dtype=np.uint8)
        
        if resize:
            resized_image, transform_info = resize_with_aspect_ratio(
                image_array,
                self.image_size,
                pad=True,
                pad_value=255
            )
        else:
            resized_image = cv2.resize(
                image_array,
                (self.image_size[1], self.image_size[0]),
                interpolation=cv2.INTER_LINEAR
            )
            transform_info = {
                'scale': self.image_size[0] / image_array.shape[0],
                'original_size': image_array.shape[:2]
            }
        
        dummy_mask = np.zeros(resized_image.shape[:2], dtype=np.int32)
        transformed_image, _ = self.transform(resized_image, dummy_mask)
        
        image_tensor = transformed_image.unsqueeze(0).to(self.device)
        
        return {
            'tensor': image_tensor,
            'original': image_array,
            'resized': resized_image,
            'original_size': original_size,
            'transform_info': transform_info,
            'path': str(image_path)
        }
    
    def predict(
        self,
        image_data: Dict[str, Any],
        apply_postprocessing: bool = True
    ) -> Dict[str, Any]:
        """
        Run inference on preprocessed image.
        
        Args:
            image_data: Preprocessed image data
            apply_postprocessing: Whether to apply post-processing
            
        Returns:
            Dictionary with predictions and metadata
        """
        with torch.no_grad():
            logits = self.model(image_data['tensor'])
            
            probabilities = torch.softmax(logits, dim=1)
            
            predicted_mask = torch.argmax(logits, dim=1)
            predicted_mask = predicted_mask.squeeze(0).cpu().numpy()
        
        if apply_postprocessing:
            refined_mask = self.post_processor.refine_mask(
                predicted_mask,
                apply_morphology=True,
                fill_holes=True,
                remove_small=True
            )
        else:
            refined_mask = predicted_mask
        
        characters = self.post_processor.extract_character_instances(
            refined_mask,
            image=image_data['resized']
        )
        
        lines = extract_line_segments(characters, line_threshold=50)
        
        spacing_stats = calculate_character_spacing(characters)
        
        return {
            'mask': predicted_mask,
            'refined_mask': refined_mask,
            'probabilities': probabilities.squeeze(0).cpu().numpy(),
            'characters': characters,
            'lines': lines,
            'spacing_stats': spacing_stats,
            'num_characters': len(characters),
            'num_lines': len(lines)
        }
    
    def process_image(
        self,
        image_path: str,
        output_dir: Optional[str] = None,
        save_visualization: bool = True,
        save_characters: bool = False,
        save_json: bool = True
    ) -> Dict[str, Any]:
        """
        Complete inference pipeline for a single image.
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save results
            save_visualization: Save visualization plots
            save_characters: Save individual character crops
            save_json: Save results as JSON
            
        Returns:
            Complete results dictionary
        """
        image_path = Path(image_path)
        
        if output_dir is None:
            output_dir = Path(self.config.get('paths.predictions_dir', 'outputs/predictions'))
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nProcessing: {image_path.name}")
        
        image_data = self.preprocess_image(str(image_path))
        
        predictions = self.predict(image_data, apply_postprocessing=True)
        
        results = {
            'image_path': str(image_path),
            'image_name': image_path.name,
            'original_size': image_data['original_size'],
            'num_characters_detected': predictions['num_characters'],
            'num_lines_detected': predictions['num_lines'],
            'spacing_statistics': {
                k: float(v) if isinstance(v, (np.floating, float)) else v
                for k, v in predictions['spacing_stats'].items()
                if k != 'spacings'
            },
            'characters': [
                {
                    'id': char['instance_id'],
                    'bbox': [int(x) for x in char['bbox']],
                    'center': [float(x) for x in char['center']],
                    'area': int(char['area']),
                    'width': int(char['width']),
                    'height': int(char['height']),
                    'aspect_ratio': float(char['aspect_ratio'])
                }
                for char in predictions['characters']
            ],
            'lines': [
                {
                    'line_id': idx,
                    'num_characters': len(line),
                    'character_ids': [char['instance_id'] for char in line],
                    'y_position': float(line[0]['center'][1]) if line else 0
                }
                for idx, line in enumerate(predictions['lines'])
            ]
        }
        
        base_name = image_path.stem
        
        if save_visualization:
            vis_path = output_dir / f"{base_name}_visualization.png"
            self.visualizer.visualize_prediction(
                image=image_data['resized'],
                ground_truth=np.zeros_like(predictions['refined_mask']),
                prediction=predictions['refined_mask'],
                title=f"Character Segmentation: {image_path.name}",
                save_path=str(vis_path)
            )
            
            bbox_image = draw_bboxes_on_image(
                image_data['resized'],
                [char['bbox'] for char in predictions['characters']],
                labels=[f"C{char['instance_id']}" for char in predictions['characters']],
                color=(0, 255, 0),
                thickness=2
            )
            bbox_path = output_dir / f"{base_name}_bboxes.png"
            cv2.imwrite(str(bbox_path), cv2.cvtColor(bbox_image, cv2.COLOR_RGB2BGR))
            
            results['visualization_path'] = str(vis_path)
            results['bbox_image_path'] = str(bbox_path)
        
        mask_path = output_dir / f"{base_name}_mask.png"
        cv2.imwrite(str(mask_path), predictions['refined_mask'].astype(np.uint16))
        results['mask_path'] = str(mask_path)
        
        if save_characters and predictions['characters']:
            chars_dir = output_dir / f"{base_name}_characters"
            save_character_crops(
                predictions['characters'],
                str(chars_dir),
                prefix=base_name
            )
            results['characters_dir'] = str(chars_dir)
        
        if save_json:
            json_path = output_dir / f"{base_name}_results.json"
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2)
            results['json_path'] = str(json_path)
        
        print(f"  Detected: {predictions['num_characters']} characters in {predictions['num_lines']} lines")
        print(f"  Results saved to: {output_dir}")
        
        return results
    
    def process_batch(
        self,
        image_paths: List[str],
        output_dir: str,
        save_visualization: bool = True,
        save_characters: bool = False,
        save_summary: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Process multiple images in batch.
        
        Args:
            image_paths: List of image paths
            output_dir: Directory to save results
            save_visualization: Save visualizations
            save_characters: Save character crops
            save_summary: Save batch summary
            
        Returns:
            List of results for each image
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nProcessing {len(image_paths)} images...")
        
        all_results = []
        
        for image_path in tqdm(image_paths, desc="Processing images"):
            try:
                results = self.process_image(
                    image_path,
                    output_dir=output_dir,
                    save_visualization=save_visualization,
                    save_characters=save_characters,
                    save_json=True
                )
                all_results.append(results)
            except Exception as e:
                print(f"\nError processing {image_path}: {e}")
                continue
        
        if save_summary:
            summary = {
                'total_images': len(image_paths),
                'successful': len(all_results),
                'failed': len(image_paths) - len(all_results),
                'total_characters_detected': sum(r['num_characters_detected'] for r in all_results),
                'total_lines_detected': sum(r['num_lines_detected'] for r in all_results),
                'average_characters_per_image': sum(r['num_characters_detected'] for r in all_results) / len(all_results) if all_results else 0,
                'timestamp': datetime.now().isoformat(),
                'results': all_results
            }
            
            summary_path = output_dir / "batch_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"\nBatch Summary:")
            print(f"  Total images: {summary['total_images']}")
            print(f"  Successful: {summary['successful']}")
            print(f"  Total characters: {summary['total_characters_detected']}")
            print(f"  Avg chars/image: {summary['average_characters_per_image']:.1f}")
            print(f"  Summary saved to: {summary_path}")
        
        return all_results


def main():
    """Main entry point for inference."""
    parser = argparse.ArgumentParser(
        description="Run inference on handwritten character segmentation model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference.py --checkpoint outputs/checkpoints/best_model.ckpt --image path/to/image.png
  python inference.py --checkpoint outputs/checkpoints/best_model.ckpt --image_dir path/to/images/
  python inference.py --checkpoint outputs/checkpoints/best_model.ckpt --image image.png --save_chars
  python inference.py --checkpoint outputs/checkpoints/best_model.ckpt --image image.png --output results/
        """
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to trained model checkpoint (.ckpt file)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--image',
        type=str,
        help='Path to single image for inference'
    )
    
    parser.add_argument(
        '--image_dir',
        type=str,
        help='Directory containing images for batch inference'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/predictions',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--save_chars',
        action='store_true',
        help='Save individual character crops'
    )
    
    parser.add_argument(
        '--no_visualization',
        action='store_true',
        help='Disable visualization generation'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu', 'auto'],
        default='auto',
        help='Device to run inference on'
    )
    
    parser.add_argument(
        '--extensions',
        nargs='+',
        default=['.png', '.jpg', '.jpeg', '.tiff', '.bmp'],
        help='Image file extensions to process'
    )
    
    args = parser.parse_args()
    
    if not args.image and not args.image_dir:
        parser.error("Either --image or --image_dir must be specified")
    
    device = None if args.device == 'auto' else args.device
    
    try:
        inference_pipeline = CharacterSegmentationInference(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            device=device
        )
        
        if args.image:
            results = inference_pipeline.process_image(
                image_path=args.image,
                output_dir=args.output,
                save_visualization=not args.no_visualization,
                save_characters=args.save_chars,
                save_json=True
            )
            
            print("\n" + "=" * 80)
            print("INFERENCE COMPLETE")
            print("=" * 80)
            print(f"Characters detected: {results['num_characters_detected']}")
            print(f"Lines detected: {results['num_lines_detected']}")
            print(f"Results saved to: {args.output}")
            print("=" * 80)
        
        elif args.image_dir:
            image_dir = Path(args.image_dir)
            
            if not image_dir.exists():
                raise FileNotFoundError(f"Image directory not found: {image_dir}")
            
            image_paths = []
            for ext in args.extensions:
                image_paths.extend(image_dir.glob(f"*{ext}"))
                image_paths.extend(image_dir.glob(f"*{ext.upper()}"))
            
            image_paths = sorted(set(str(p) for p in image_paths))
            
            if not image_paths:
                print(f"No images found in {image_dir}")
                return
            
            print(f"Found {len(image_paths)} images")
            
            results = inference_pipeline.process_batch(
                image_paths=image_paths,
                output_dir=args.output,
                save_visualization=not args.no_visualization,
                save_characters=args.save_chars,
                save_summary=True
            )
            
            print("\n" + "=" * 80)
            print("BATCH INFERENCE COMPLETE")
            print("=" * 80)
            print(f"Processed: {len(results)} images")
            print(f"Results saved to: {args.output}")
            print("=" * 80)
    
    except Exception as e:
        print(f"\nInference failed: {e}")
        raise


if __name__ == "__main__":
    main()