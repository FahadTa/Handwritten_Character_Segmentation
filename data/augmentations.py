"""
Augmentation pipeline for handwritten character segmentation.
Applies realistic document transformations while preserving mask alignment.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Dict, Any
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SegmentationAugmentation:
    """
    Augmentation pipeline that applies transformations to both image and mask.
    Ensures perfect alignment between augmented image and segmentation mask.
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (1024, 1024),
        augmentation_config: Optional[Dict[str, Any]] = None,
        training: bool = True
    ):
        """
        Initialize augmentation pipeline.
        
        Args:
            image_size: Target (height, width)
            augmentation_config: Configuration dict from config.yaml
            training: Whether to apply training augmentations
        """
        self.image_size = image_size
        self.config = augmentation_config or {}
        self.training = training
        
        self.train_transform = self._build_train_augmentations()
        self.val_transform = self._build_val_augmentations()
    
    def _build_train_augmentations(self) -> A.Compose:
        """
        Build training augmentation pipeline with realistic document variations.
        
        Returns:
            Albumentations Compose object
        """
        geometric_config = self.config.get('geometric', {})
        photometric_config = self.config.get('photometric', {})
        noise_blur_config = self.config.get('noise_blur', {})
        
        augmentations = []
        
        if self.config.get('enabled', True):
            prob = self.config.get('probability', 0.8)
            
            augmentations.extend([
                A.Rotate(
                    limit=geometric_config.get('rotation_range', [-5, 5]),
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=255,
                    mask_value=0,
                    p=prob
                ),
                
                A.ShiftScaleRotate(
                    shift_limit=geometric_config.get('translate_percent', 0.05),
                    scale_limit=[
                        geometric_config.get('scale_range', [0.95, 1.05])[0] - 1,
                        geometric_config.get('scale_range', [0.95, 1.05])[1] - 1
                    ],
                    rotate_limit=0,
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=255,
                    mask_value=0,
                    p=prob
                ),
                
                A.ElasticTransform(
                    alpha=geometric_config.get('elastic_transform', {}).get('alpha', 50),
                    sigma=geometric_config.get('elastic_transform', {}).get('sigma', 5),
                    alpha_affine=0,
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=255,
                    mask_value=0,
                    p=prob * 0.3
                ),
                
                A.Perspective(
                    scale=geometric_config.get('shear_range', [-5, 5])[1] / 100,
                    keep_size=True,
                    pad_mode=cv2.BORDER_CONSTANT,
                    pad_val=255,
                    mask_pad_val=0,
                    fit_output=False,
                    interpolation=cv2.INTER_LINEAR,
                    p=prob * 0.5
                ),
                
                A.OneOf([
                    A.RandomBrightnessContrast(
                        brightness_limit=self._get_limit(
                            photometric_config.get('brightness_range', [0.8, 1.2])
                        ),
                        contrast_limit=self._get_limit(
                            photometric_config.get('contrast_range', [0.8, 1.2])
                        ),
                        brightness_by_max=False,
                        p=1.0
                    ),
                    A.RandomGamma(
                        gamma_limit=self._get_gamma_limit(
                            photometric_config.get('gamma_range', [0.9, 1.1])
                        ),
                        p=1.0
                    ),
                ], p=prob),
                
                A.OneOf([
                    A.GaussNoise(
                        var_limit=self._scale_noise(
                            noise_blur_config.get('gaussian_noise_var', [0, 0.01])
                        ),
                        mean=0,
                        per_channel=False,
                        p=1.0
                    ),
                    A.GaussianBlur(
                        blur_limit=(3, 7),
                        sigma_limit=tuple(noise_blur_config.get('gaussian_blur_sigma', [0, 1.5])),
                        p=1.0
                    ),
                    A.MotionBlur(
                        blur_limit=tuple(noise_blur_config.get('motion_blur_kernel', [3, 7])),
                        p=1.0
                    ),
                ], p=prob * 0.5),
                
                A.CoarseDropout(
                    max_holes=8,
                    max_height=32,
                    max_width=32,
                    min_holes=1,
                    min_height=8,
                    min_width=8,
                    fill_value=255,
                    mask_fill_value=0,
                    p=prob * 0.3
                ),
            ])
        
        augmentations.append(A.Resize(
            height=self.image_size[0],
            width=self.image_size[1],
            interpolation=cv2.INTER_LINEAR,
            p=1.0
        ))
        
        augmentations.append(A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0
        ))
        
        augmentations.append(ToTensorV2())
        
        return A.Compose(
            augmentations,
            additional_targets={'mask': 'mask'}
        )
    
    def _build_val_augmentations(self) -> A.Compose:
        """
        Build validation/test augmentation pipeline (minimal transforms).
        
        Returns:
            Albumentations Compose object
        """
        return A.Compose([
            A.Resize(
                height=self.image_size[0],
                width=self.image_size[1],
                interpolation=cv2.INTER_LINEAR,
                p=1.0
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
            ToTensorV2()
        ], additional_targets={'mask': 'mask'})
    
    def _get_limit(self, value_range: list) -> Tuple[float, float]:
        """
        Convert multiplicative range to additive limit for albumentations.
        
        Args:
            value_range: [min_multiplier, max_multiplier], e.g., [0.8, 1.2]
            
        Returns:
            Tuple of (lower_limit, upper_limit)
        """
        lower = value_range[0] - 1.0
        upper = value_range[1] - 1.0
        return (lower, upper)
    
    def _get_gamma_limit(self, gamma_range: list) -> Tuple[int, int]:
        """
        Convert gamma range to integer limits for albumentations.
        
        Args:
            gamma_range: [min_gamma, max_gamma]
            
        Returns:
            Tuple of integer limits scaled by 100
        """
        return (int(gamma_range[0] * 100), int(gamma_range[1] * 100))
    
    def _scale_noise(self, var_range: list) -> Tuple[float, float]:
        """
        Scale noise variance range to pixel values.
        
        Args:
            var_range: Variance range [min, max]
            
        Returns:
            Scaled variance range
        """
        return (var_range[0] * 255 * 255, var_range[1] * 255 * 255)
    
    def __call__(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[Any, Any]:
        """
        Apply augmentations to image and mask.
        
        Args:
            image: Input image (H, W, 3) as uint8
            mask: Segmentation mask (H, W) as int32
            
        Returns:
            Tuple of (augmented_image_tensor, augmented_mask_tensor)
        """
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        if mask.dtype != np.int32:
            mask = mask.astype(np.int32)
        
        transform = self.train_transform if self.training else self.val_transform
        
        transformed = transform(image=image, mask=mask)
        
        return transformed['image'], transformed['mask']


class DocumentAugmentation:
    """
    Document-specific augmentations for realistic handwriting simulation.
    These are applied as pre-processing before the main augmentation pipeline.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize document augmentation.
        
        Args:
            config: Document augmentation configuration
        """
        self.config = config or {}
    
    def apply_paper_texture(
        self,
        image: np.ndarray,
        intensity: float = 0.1
    ) -> np.ndarray:
        """
        Add subtle paper texture to simulate real document.
        
        Args:
            image: Input image
            intensity: Texture intensity (0-1)
            
        Returns:
            Image with paper texture
        """
        if not self.config.get('paper_texture', False):
            return image
        
        noise = np.random.normal(0, intensity * 10, image.shape[:2])
        
        texture = cv2.GaussianBlur(noise, (5, 5), 0)
        texture = ((texture - texture.min()) / (texture.max() - texture.min()) * 255).astype(np.uint8)
        
        if image.ndim == 3:
            texture = cv2.cvtColor(texture, cv2.COLOR_GRAY2RGB)
        
        result = cv2.addWeighted(image, 0.95, texture, 0.05, 0)
        
        return result
    
    def apply_ink_bleed(
        self,
        image: np.ndarray,
        kernel_size: int = 3
    ) -> np.ndarray:
        """
        Simulate ink bleeding effect on text.
        
        Args:
            image: Input image
            kernel_size: Dilation kernel size
            
        Returns:
            Image with ink bleed effect
        """
        if not self.config.get('ink_bleed', False):
            return image
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        
        text_mask = (gray < 200).astype(np.uint8)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated = cv2.dilate(text_mask, kernel, iterations=1)
        
        if image.ndim == 3:
            result = image.copy()
            for i in range(3):
                result[:, :, i] = np.where(dilated > 0, image[:, :, i] * 0.95, image[:, :, i])
        else:
            result = np.where(dilated > 0, image * 0.95, image)
        
        return result.astype(np.uint8)
    
    def apply_shadow(
        self,
        image: np.ndarray,
        intensity: float = 0.3
    ) -> np.ndarray:
        """
        Add subtle shadow gradient to simulate lighting.
        
        Args:
            image: Input image
            intensity: Shadow intensity (0-1)
            
        Returns:
            Image with shadow
        """
        if not self.config.get('shadow', False):
            return image
        
        h, w = image.shape[:2]
        
        x = np.linspace(0, 1, w)
        y = np.linspace(0, 1, h)
        X, Y = np.meshgrid(x, y)
        
        shadow = 1 - (intensity * (X + Y) / 2)
        shadow = np.clip(shadow, 1 - intensity, 1.0)
        
        if image.ndim == 3:
            shadow = shadow[:, :, np.newaxis]
        
        result = (image * shadow).astype(np.uint8)
        
        return result
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply all enabled document augmentations.
        
        Args:
            image: Input image
            
        Returns:
            Augmented image
        """
        image = self.apply_paper_texture(image)
        image = self.apply_ink_bleed(image)
        image = self.apply_shadow(image)
        
        return image


def get_augmentation_pipeline(
    config: Dict[str, Any],
    training: bool = True
) -> SegmentationAugmentation:
    """
    Factory function to create augmentation pipeline from config.
    
    Args:
        config: Configuration dictionary
        training: Whether for training or validation
        
    Returns:
        SegmentationAugmentation instance
    """
    image_size = tuple(config.get('dataset_generation', {}).get('image_size', [1024, 1024]))
    aug_config = config.get('augmentation', {})
    
    return SegmentationAugmentation(
        image_size=image_size,
        augmentation_config=aug_config,
        training=training
    )