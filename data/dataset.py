"""
PyTorch Dataset for handwritten character segmentation.
Loads images, masks, and applies augmentations.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image
import torch
from torch.utils.data import Dataset

from data.augmentations import SegmentationAugmentation, DocumentAugmentation


class CharacterSegmentationDataset(Dataset):
    """
    Dataset for character-level instance segmentation.
    Loads synthetic handwritten text images with corresponding instance masks.
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[SegmentationAugmentation] = None,
        document_augmentation: Optional[DocumentAugmentation] = None,
        cache_data: bool = False
    ):
        """
        Initialize dataset.
        
        Args:
            data_dir: Root directory containing images/, masks/, metadata/
            split: One of 'train', 'val', 'test'
            transform: Augmentation pipeline
            document_augmentation: Document-specific augmentations
            cache_data: Whether to cache loaded data in memory
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.document_augmentation = document_augmentation
        self.cache_data = cache_data
        
        self.images_dir = self.data_dir / "images"
        self.masks_dir = self.data_dir / "masks"
        self.metadata_dir = self.data_dir / "metadata"
        
        self._verify_directories()
        
        self.samples = self._load_split()
        
        if self.cache_data:
            self.cache = {}
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
    
    def _verify_directories(self) -> None:
        """Verify all required directories exist."""
        required_dirs = [self.images_dir, self.masks_dir, self.metadata_dir]
        
        for directory in required_dirs:
            if not directory.exists():
                raise FileNotFoundError(
                    f"Required directory not found: {directory}\n"
                    f"Please run generate_dataset.py first to create the dataset."
                )
    
    def _load_split(self) -> List[str]:
        """
        Load sample IDs for the specified split.
        
        Returns:
            List of sample IDs
        """
        splits_file = self.data_dir / "dataset_splits.json"
        
        if not splits_file.exists():
            raise FileNotFoundError(
                f"Dataset splits file not found: {splits_file}\n"
                f"Please run generate_dataset.py to create splits."
            )
        
        with open(splits_file, 'r') as f:
            splits = json.load(f)
        
        if self.split not in splits:
            raise ValueError(
                f"Split '{self.split}' not found in dataset splits.\n"
                f"Available splits: {list(splits.keys())}"
            )
        
        sample_ids = splits[self.split]['image_ids']
        
        return sample_ids
    
    def _load_image(self, sample_id: str) -> np.ndarray:
        """
        Load image from disk.
        
        Args:
            sample_id: Sample identifier
            
        Returns:
            Image as RGB numpy array (H, W, 3)
        """
        image_path = self.images_dir / f"{sample_id}.png"
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(image_path)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = np.array(image, dtype=np.uint8)
        
        return image
    
    def _load_mask(self, sample_id: str) -> np.ndarray:
        """
        Load instance segmentation mask from disk.
        
        Args:
            sample_id: Sample identifier
            
        Returns:
            Mask as numpy array (H, W) with instance IDs
        """
        mask_path = self.masks_dir / f"{sample_id}_mask.png"
        
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        
        mask = Image.open(mask_path)
        mask = np.array(mask, dtype=np.int32)
        
        return mask
    
    def _load_metadata(self, sample_id: str) -> Dict[str, Any]:
        """
        Load sample metadata.
        
        Args:
            sample_id: Sample identifier
            
        Returns:
            Metadata dictionary
        """
        metadata_path = self.metadata_dir / f"{sample_id}.json"
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return metadata
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing:
                - image: Tensor (3, H, W)
                - mask: Tensor (H, W)
                - sample_id: str
                - metadata: Dict (optional)
        """
        sample_id = self.samples[idx]
        
        if self.cache_data and sample_id in self.cache:
            image, mask, metadata = self.cache[sample_id]
        else:
            image = self._load_image(sample_id)
            mask = self._load_mask(sample_id)
            metadata = self._load_metadata(sample_id)
            
            if self.cache_data:
                self.cache[sample_id] = (image.copy(), mask.copy(), metadata)
        
        if self.document_augmentation is not None:
            image = self.document_augmentation(image)
        
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()
        
        return {
            'image': image,
            'mask': mask,
            'sample_id': sample_id,
            'num_characters': metadata['num_characters'],
            'text_content': metadata['text_content']
        }
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """
        Compute dataset statistics.
        
        Returns:
            Dictionary with statistics
        """
        total_chars = 0
        total_instances = 0
        char_lengths = []
        
        for sample_id in self.samples:
            metadata = self._load_metadata(sample_id)
            num_chars = metadata['num_characters']
            total_chars += num_chars
            char_lengths.append(num_chars)
            
            mask = self._load_mask(sample_id)
            num_instances = len(np.unique(mask)) - 1
            total_instances += num_instances
        
        stats = {
            'num_samples': len(self.samples),
            'total_characters': total_chars,
            'total_instances': total_instances,
            'avg_chars_per_image': total_chars / len(self.samples) if self.samples else 0,
            'avg_instances_per_image': total_instances / len(self.samples) if self.samples else 0,
            'min_chars': min(char_lengths) if char_lengths else 0,
            'max_chars': max(char_lengths) if char_lengths else 0
        }
        
        return stats
    
    def visualize_sample(
        self,
        idx: int,
        denormalize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Get a sample for visualization.
        
        Args:
            idx: Sample index
            denormalize: Whether to denormalize the image
            
        Returns:
            Tuple of (image, mask, metadata)
        """
        sample = self[idx]
        
        image = sample['image']
        mask = sample['mask']
        
        if isinstance(image, torch.Tensor):
            image = image.numpy()
            
            if denormalize:
                mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
                std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
                image = (image * std + mean) * 255
                image = np.clip(image, 0, 255).astype(np.uint8)
            else:
                image = (image * 255).astype(np.uint8)
            
            image = np.transpose(image, (1, 2, 0))
        
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()
        
        metadata = {
            'sample_id': sample['sample_id'],
            'num_characters': sample['num_characters'],
            'text_content': sample['text_content']
        }
        
        return image, mask, metadata


class CharacterSegmentationInferenceDataset(Dataset):
    """
    Dataset for inference on real handwritten documents.
    Loads images without requiring masks.
    """
    
    def __init__(
        self,
        image_paths: List[str],
        transform: Optional[SegmentationAugmentation] = None,
        image_size: Tuple[int, int] = (1024, 1024)
    ):
        """
        Initialize inference dataset.
        
        Args:
            image_paths: List of paths to images
            transform: Augmentation pipeline (validation transforms)
            image_size: Target image size
        """
        self.image_paths = [Path(p) for p in image_paths]
        self.transform = transform
        self.image_size = image_size
        
        self._verify_images()
        
        print(f"Loaded {len(self.image_paths)} images for inference")
    
    def _verify_images(self) -> None:
        """Verify all image files exist."""
        missing = [p for p in self.image_paths if not p.exists()]
        
        if missing:
            raise FileNotFoundError(
                f"The following image files were not found:\n" +
                "\n".join(str(p) for p in missing)
            )
    
    def __len__(self) -> int:
        """Return number of images."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single image for inference.
        
        Args:
            idx: Image index
            
        Returns:
            Dictionary containing:
                - image: Tensor (3, H, W)
                - image_path: str
                - original_size: Tuple[int, int]
        """
        image_path = self.image_paths[idx]
        
        image = Image.open(image_path)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        original_size = image.size
        
        image = np.array(image, dtype=np.uint8)
        
        if self.transform is not None:
            dummy_mask = np.zeros(image.shape[:2], dtype=np.int32)
            image, _ = self.transform(image, dummy_mask)
        else:
            import cv2
            image = cv2.resize(image, self.image_size[::-1])
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return {
            'image': image,
            'image_path': str(image_path),
            'original_size': original_size
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for batching samples.
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        Batched dictionary
    """
    images = torch.stack([item['image'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])
    
    sample_ids = [item['sample_id'] for item in batch]
    num_characters = [item['num_characters'] for item in batch]
    text_contents = [item['text_content'] for item in batch]
    
    return {
        'image': images,
        'mask': masks,
        'sample_id': sample_ids,
        'num_characters': num_characters,
        'text_content': text_contents
    }


def get_dataset(
    data_dir: str,
    split: str,
    config: Dict[str, Any],
    training: bool = True
) -> CharacterSegmentationDataset:
    """
    Factory function to create dataset from configuration.
    
    Args:
        data_dir: Data directory path
        split: Dataset split ('train', 'val', 'test')
        config: Configuration dictionary
        training: Whether this is for training
        
    Returns:
        CharacterSegmentationDataset instance
    """
    from data.augmentations import get_augmentation_pipeline
    
    transform = get_augmentation_pipeline(config, training=training)
    
    doc_aug_config = config.get('augmentation', {}).get('document', {})
    document_augmentation = DocumentAugmentation(doc_aug_config) if training else None
    
    dataset = CharacterSegmentationDataset(
        data_dir=data_dir,
        split=split,
        transform=transform,
        document_augmentation=document_augmentation,
        cache_data=False
    )
    
    return dataset