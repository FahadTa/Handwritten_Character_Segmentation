"""
PyTorch Lightning module for handwritten character segmentation training.
Handles training/validation/test loops, optimization, metrics, and logging.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    StepLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    OneCycleLR
)
import pytorch_lightning as pl
from typing import Dict, Any, Optional, List
import wandb

from models import (
    create_unet,
    get_loss_function,
    SegmentationMetrics,
    compute_iou,
    compute_dice,
    compute_pixel_accuracy
)


class CharacterSegmentationModule(pl.LightningModule):
    """
    Lightning module for training character segmentation models.
    
    Handles:
    - Model initialization and forward pass
    - Loss computation with combined objectives
    - Metric tracking (IoU, Dice, Accuracy, etc.)
    - Optimizer and scheduler configuration
    - Logging to Weights & Biases
    - Gradient clipping for stable training
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Lightning module.
        
        Args:
            config: Configuration dictionary containing all hyperparameters
        """
        super(CharacterSegmentationModule, self).__init__()
        
        self.config = config
        self.save_hyperparameters(config)
        
        self.model = create_unet(config)
        
        self.num_classes = config.get('model', {}).get('unet', {}).get('num_classes', 64)
        
        self.loss_fn = None
        
        self.train_metrics = SegmentationMetrics(
            num_classes=self.num_classes,
            ignore_index=-100
        )
        self.val_metrics = SegmentationMetrics(
            num_classes=self.num_classes,
            ignore_index=-100
        )
        self.test_metrics = SegmentationMetrics(
            num_classes=self.num_classes,
            ignore_index=-100
        )
        
        self.use_amp = config.get('training', {}).get('use_amp', True)
        
        self.gradient_clip_val = config.get('training', {}).get('gradient_clip_val', 1.0)
        
        self.log_images = config.get('logging', {}).get('wandb', {}).get('log_images', True)
        self.log_frequency = config.get('logging', {}).get('wandb', {}).get('log_frequency', 100)
        self.num_images_to_log = config.get('logging', {}).get('wandb', {}).get('num_images_to_log', 4)
        
        self.validation_step_outputs = []
        self.test_step_outputs = []
    
    def setup(self, stage: Optional[str] = None):
        """
        Setup hook called before training/validation/test.
        Initialize loss function here to ensure it's on the correct device.
        
        Args:
            stage: Current stage ('fit', 'validate', 'test', 'predict')
        """
        if self.loss_fn is None:
            self.loss_fn = get_loss_function(self.config, self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through model.
        
        Args:
            x: Input images (B, 3, H, W)
            
        Returns:
            Segmentation logits (B, num_classes, H, W)
        """
        return self.model(x)
    
    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute segmentation loss.
        
        Args:
            predictions: Model predictions (B, C, H, W)
            targets: Ground truth masks (B, H, W)
            
        Returns:
            Loss value
        """
        return self.loss_fn(predictions, targets)
    
    def training_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int
    ) -> torch.Tensor:
        """
        Training step for one batch.
        
        Args:
            batch: Batch dictionary with 'image' and 'mask'
            batch_idx: Batch index
            
        Returns:
            Loss value
        """
        images = batch['image']
        masks = batch['mask']
        
        predictions = self(images)
        
        loss = self.compute_loss(predictions, masks)
        
        with torch.no_grad():
            iou = compute_iou(predictions, masks, self.num_classes)
            dice = compute_dice(predictions, masks, self.num_classes)
            pixel_acc = compute_pixel_accuracy(predictions, masks)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_iou', iou, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_dice', dice, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('train_pixel_acc', pixel_acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        
        if self.log_images and batch_idx % self.log_frequency == 0:
            self._log_images(images, masks, predictions, prefix='train', max_images=2)
        
        return loss
    
    def validation_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Validation step for one batch.
        
        Args:
            batch: Batch dictionary with 'image' and 'mask'
            batch_idx: Batch index
            
        Returns:
            Dictionary with loss and metrics
        """
        images = batch['image']
        masks = batch['mask']
        
        predictions = self(images)
        
        loss = self.compute_loss(predictions, masks)
        
        iou = compute_iou(predictions, masks, self.num_classes)
        dice = compute_dice(predictions, masks, self.num_classes)
        pixel_acc = compute_pixel_accuracy(predictions, masks)
        
        self.val_metrics.update(predictions, masks)
        
        output = {
            'val_loss': loss,
            'val_iou': iou,
            'val_dice': dice,
            'val_pixel_acc': pixel_acc
        }
        
        self.validation_step_outputs.append(output)
        
        if self.log_images and batch_idx == 0:
            self._log_images(images, masks, predictions, prefix='val', max_images=self.num_images_to_log)
        
        return output
    
    def on_validation_epoch_end(self):
        """Aggregate validation metrics at epoch end."""
        if len(self.validation_step_outputs) == 0:
            return
        
        avg_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()
        avg_iou = torch.stack([x['val_iou'] for x in self.validation_step_outputs]).mean()
        avg_dice = torch.stack([x['val_dice'] for x in self.validation_step_outputs]).mean()
        avg_pixel_acc = torch.stack([x['val_pixel_acc'] for x in self.validation_step_outputs]).mean()
        
        self.log('val_loss_epoch', avg_loss, prog_bar=True, logger=True)
        self.log('val_iou_epoch', avg_iou, prog_bar=True, logger=True)
        self.log('val_dice_epoch', avg_dice, prog_bar=True, logger=True)
        self.log('val_pixel_acc_epoch', avg_pixel_acc, prog_bar=False, logger=True)
        
        all_metrics = self.val_metrics.compute_all()
        for metric_name, metric_value in all_metrics.items():
            self.log(f'val_{metric_name}_detailed', metric_value, logger=True)
        
        self.val_metrics.reset()
        self.validation_step_outputs.clear()
    
    def test_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Test step for one batch.
        
        Args:
            batch: Batch dictionary with 'image' and 'mask'
            batch_idx: Batch index
            
        Returns:
            Dictionary with loss and metrics
        """
        images = batch['image']
        masks = batch['mask']
        
        predictions = self(images)
        
        loss = self.compute_loss(predictions, masks)
        
        iou = compute_iou(predictions, masks, self.num_classes)
        dice = compute_dice(predictions, masks, self.num_classes)
        pixel_acc = compute_pixel_accuracy(predictions, masks)
        
        self.test_metrics.update(predictions, masks)
        
        output = {
            'test_loss': loss,
            'test_iou': iou,
            'test_dice': dice,
            'test_pixel_acc': pixel_acc
        }
        
        self.test_step_outputs.append(output)
        
        if self.log_images and batch_idx < 5:
            self._log_images(images, masks, predictions, prefix='test', max_images=4)
        
        return output
    
    def on_test_epoch_end(self):
        """Aggregate test metrics at epoch end."""
        if len(self.test_step_outputs) == 0:
            return
        
        avg_loss = torch.stack([x['test_loss'] for x in self.test_step_outputs]).mean()
        avg_iou = torch.stack([x['test_iou'] for x in self.test_step_outputs]).mean()
        avg_dice = torch.stack([x['test_dice'] for x in self.test_step_outputs]).mean()
        avg_pixel_acc = torch.stack([x['test_pixel_acc'] for x in self.test_step_outputs]).mean()
        
        self.log('test_loss', avg_loss, logger=True)
        self.log('test_iou', avg_iou, logger=True)
        self.log('test_dice', avg_dice, logger=True)
        self.log('test_pixel_acc', avg_pixel_acc, logger=True)
        
        all_metrics = self.test_metrics.compute_all()
        for metric_name, metric_value in all_metrics.items():
            self.log(f'test_{metric_name}', metric_value, logger=True)
        
        print("\n" + "=" * 60)
        print("TEST RESULTS")
        print("=" * 60)
        print(f"Loss: {avg_loss:.4f}")
        print(f"IoU: {avg_iou:.4f}")
        print(f"Dice: {avg_dice:.4f}")
        print(f"Pixel Accuracy: {avg_pixel_acc:.4f}")
        print(f"Precision: {all_metrics['precision']:.4f}")
        print(f"Recall: {all_metrics['recall']:.4f}")
        print(f"F1: {all_metrics['f1']:.4f}")
        print("=" * 60 + "\n")
        
        self.test_metrics.reset()
        self.test_step_outputs.clear()
    
    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.
        
        Returns:
            Optimizer and scheduler configuration
        """
        optimizer_config = self.config.get('training', {}).get('optimizer', {})
        scheduler_config = self.config.get('training', {}).get('scheduler', {})
        
        optimizer_name = optimizer_config.get('name', 'adamw').lower()
        lr = optimizer_config.get('learning_rate', 0.001)
        weight_decay = optimizer_config.get('weight_decay', 0.0001)
        
        if optimizer_name == 'adam':
            optimizer = optim.Adam(
                self.parameters(),
                lr=lr,
                betas=tuple(optimizer_config.get('betas', [0.9, 0.999])),
                weight_decay=weight_decay
            )
        elif optimizer_name == 'adamw':
            optimizer = optim.AdamW(
                self.parameters(),
                lr=lr,
                betas=tuple(optimizer_config.get('betas', [0.9, 0.999])),
                weight_decay=weight_decay
            )
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(
                self.parameters(),
                lr=lr,
                momentum=optimizer_config.get('momentum', 0.9),
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        scheduler_name = scheduler_config.get('name', 'cosine').lower()
        
        if scheduler_name == 'step':
            scheduler = StepLR(
                optimizer,
                step_size=scheduler_config.get('step_size', 30),
                gamma=scheduler_config.get('gamma', 0.1)
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        
        elif scheduler_name == 'cosine':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.config.get('training', {}).get('num_epochs', 100),
                eta_min=scheduler_config.get('min_lr', 1e-6)
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        
        elif scheduler_name == 'plateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=scheduler_config.get('gamma', 0.1),
                patience=scheduler_config.get('patience', 10),
                min_lr=scheduler_config.get('min_lr', 1e-6)
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_iou_epoch',
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        
        else:
            return optimizer
    
    def _log_images(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        predictions: torch.Tensor,
        prefix: str = 'train',
        max_images: int = 4
    ):
        """
        Log images to Weights & Biases.
        
        Args:
            images: Input images (B, 3, H, W)
            masks: Ground truth masks (B, H, W)
            predictions: Predicted logits (B, C, H, W)
            prefix: Logging prefix ('train', 'val', 'test')
            max_images: Maximum number of images to log
        """
        if not self.logger or not hasattr(self.logger, 'experiment'):
            return
        
        try:
            pred_masks = torch.argmax(predictions, dim=1)
            
            num_images = min(max_images, images.shape[0])
            
            wandb_images = []
            
            for i in range(num_images):
                img = images[i].cpu().permute(1, 2, 0).numpy()
                img = (img * 0.229 + 0.485).clip(0, 1)
                
                mask = masks[i].cpu().numpy()
                pred_mask = pred_masks[i].cpu().numpy()
                
                wandb_images.append(
                    wandb.Image(
                        img,
                        masks={
                            "ground_truth": {
                                "mask_data": mask,
                                "class_labels": {i: f"class_{i}" for i in range(self.num_classes)}
                            },
                            "prediction": {
                                "mask_data": pred_mask,
                                "class_labels": {i: f"class_{i}" for i in range(self.num_classes)}
                            }
                        }
                    )
                )
            
            self.logger.experiment.log({
                f"{prefix}_predictions": wandb_images,
                "global_step": self.global_step
            })
        
        except Exception as e:
            print(f"Warning: Failed to log images: {e}")
    
    def on_train_epoch_end(self):
        """Hook called at the end of training epoch."""
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log('learning_rate', current_lr, logger=True)