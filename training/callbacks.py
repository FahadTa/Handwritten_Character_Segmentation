"""
Custom callbacks for training handwritten character segmentation models.
Provides checkpointing, early stopping, learning rate monitoring, and logging.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    Callback
)
import wandb


class CustomModelCheckpoint(ModelCheckpoint):
    """
    Extended ModelCheckpoint with additional features for handwriting segmentation.
    Saves models based on validation IoU and includes detailed metadata.
    """
    
    def __init__(
        self,
        dirpath: str,
        monitor: str = 'val_iou_epoch',
        mode: str = 'max',
        save_top_k: int = 3,
        save_last: bool = True,
        filename: str = 'epoch={epoch:02d}-val_iou={val_iou_epoch:.4f}',
        verbose: bool = True,
        **kwargs
    ):
        """
        Initialize custom checkpoint callback.
        
        Args:
            dirpath: Directory to save checkpoints
            monitor: Metric to monitor for checkpointing
            mode: 'min' or 'max' for the monitored metric
            save_top_k: Number of best models to keep
            save_last: Whether to save the last checkpoint
            filename: Checkpoint filename pattern
            verbose: Whether to print checkpoint messages
        """
        super().__init__(
            dirpath=dirpath,
            monitor=monitor,
            mode=mode,
            save_top_k=save_top_k,
            save_last=save_last,
            filename=filename,
            verbose=verbose,
            auto_insert_metric_name=False,
            **kwargs
        )
    
    def on_save_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Add custom metadata to checkpoint.
        
        Args:
            trainer: PyTorch Lightning trainer
            pl_module: Lightning module being trained
            checkpoint: Checkpoint dictionary
            
        Returns:
            Modified checkpoint with metadata
        """
        checkpoint['num_parameters'] = sum(
            p.numel() for p in pl_module.parameters() if p.requires_grad
        )
        checkpoint['model_architecture'] = 'UNet'
        checkpoint['num_classes'] = pl_module.num_classes
        
        return checkpoint


class CustomEarlyStopping(EarlyStopping):
    """
    Extended EarlyStopping with handwriting-specific patience strategies.
    More lenient early stopping for complex handwriting patterns.
    """
    
    def __init__(
        self,
        monitor: str = 'val_iou_epoch',
        patience: int = 15,
        mode: str = 'max',
        min_delta: float = 0.001,
        verbose: bool = True,
        **kwargs
    ):
        """
        Initialize early stopping callback.
        
        Args:
            monitor: Metric to monitor
            patience: Number of epochs with no improvement before stopping
            mode: 'min' or 'max' for the monitored metric
            min_delta: Minimum change to qualify as an improvement
            verbose: Whether to print early stopping messages
        """
        super().__init__(
            monitor=monitor,
            patience=patience,
            mode=mode,
            min_delta=min_delta,
            verbose=verbose,
            **kwargs
        )
    
    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """
        Check early stopping condition and log status.
        
        Args:
            trainer: PyTorch Lightning trainer
            pl_module: Lightning module being trained
        """
        super().on_validation_end(trainer, pl_module)
        
        if self.verbose and trainer.current_epoch > 0:
            epochs_no_improve = self.wait_count
            print(f"\nEarly Stopping: {epochs_no_improve}/{self.patience} epochs without improvement")


class MetricLogger(Callback):
    """
    Custom callback for logging additional metrics and statistics.
    Tracks training progress specific to handwriting segmentation.
    """
    
    def __init__(self, log_every_n_epochs: int = 1):
        """
        Initialize metric logger.
        
        Args:
            log_every_n_epochs: Frequency of detailed logging
        """
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
    
    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule
    ):
        """
        Log detailed training statistics at epoch end.
        
        Args:
            trainer: PyTorch Lightning trainer
            pl_module: Lightning module being trained
        """
        if trainer.current_epoch % self.log_every_n_epochs == 0:
            epoch = trainer.current_epoch
            
            train_loss = trainer.callback_metrics.get('train_loss_epoch', 0)
            train_iou = trainer.callback_metrics.get('train_iou_epoch', 0)
            val_loss = trainer.callback_metrics.get('val_loss_epoch', 0)
            val_iou = trainer.callback_metrics.get('val_iou_epoch', 0)
            
            print("\n" + "=" * 80)
            print(f"Epoch {epoch} Summary")
            print("=" * 80)
            print(f"Train Loss: {train_loss:.4f} | Train IoU: {train_iou:.4f}")
            print(f"Val Loss:   {val_loss:.4f} | Val IoU:   {val_iou:.4f}")
            print("=" * 80 + "\n")
    
    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule
    ):
        """
        Log validation metrics with detailed breakdown.
        
        Args:
            trainer: PyTorch Lightning trainer
            pl_module: Lightning module being trained
        """
        if not hasattr(pl_module.logger, 'experiment'):
            return
        
        metrics = trainer.callback_metrics
        
        detailed_metrics = {
            'epoch': trainer.current_epoch,
            'val_iou': metrics.get('val_iou_epoch', 0).item() if torch.is_tensor(metrics.get('val_iou_epoch', 0)) else metrics.get('val_iou_epoch', 0),
            'val_dice': metrics.get('val_dice_epoch', 0).item() if torch.is_tensor(metrics.get('val_dice_epoch', 0)) else metrics.get('val_dice_epoch', 0),
            'val_pixel_acc': metrics.get('val_pixel_acc_epoch', 0).item() if torch.is_tensor(metrics.get('val_pixel_acc_epoch', 0)) else metrics.get('val_pixel_acc_epoch', 0),
        }


class GradientLoggingCallback(Callback):
    """
    Callback for monitoring gradient statistics during training.
    Critical for detecting training instabilities with handwriting data.
    """
    
    def __init__(self, log_every_n_steps: int = 100):
        """
        Initialize gradient logging callback.
        
        Args:
            log_every_n_steps: Frequency of gradient logging
        """
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
    
    def on_after_backward(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule
    ):
        """
        Log gradient statistics after backward pass.
        
        Args:
            trainer: PyTorch Lightning trainer
            pl_module: Lightning module being trained
        """
        if trainer.global_step % self.log_every_n_steps == 0:
            grad_norms = []
            
            for name, param in pl_module.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    grad_norms.append(grad_norm)
            
            if grad_norms:
                avg_grad_norm = sum(grad_norms) / len(grad_norms)
                max_grad_norm = max(grad_norms)
                
                pl_module.log('train_avg_grad_norm', avg_grad_norm, on_step=True, logger=True)
                pl_module.log('train_max_grad_norm', max_grad_norm, on_step=True, logger=True)


class MemoryLoggingCallback(Callback):
    """
    Callback for monitoring GPU memory usage during training.
    Helps optimize batch size and detect memory leaks.
    """
    
    def __init__(self, log_every_n_epochs: int = 1):
        """
        Initialize memory logging callback.
        
        Args:
            log_every_n_epochs: Frequency of memory logging
        """
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
    
    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule
    ):
        """
        Log memory usage at epoch end.
        
        Args:
            trainer: PyTorch Lightning trainer
            pl_module: Lightning module being trained
        """
        if trainer.current_epoch % self.log_every_n_epochs == 0:
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                
                pl_module.log('memory_allocated_gb', memory_allocated, logger=True)
                pl_module.log('memory_reserved_gb', memory_reserved, logger=True)
                
                print(f"\nGPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")


class SavePredictionsCallback(Callback):
    """
    Callback for saving sample predictions during training.
    Useful for visual inspection of model progress on handwriting.
    """
    
    def __init__(
        self,
        output_dir: str,
        save_every_n_epochs: int = 5,
        num_samples: int = 4
    ):
        """
        Initialize predictions saving callback.
        
        Args:
            output_dir: Directory to save predictions
            save_every_n_epochs: Frequency of saving predictions
            num_samples: Number of samples to save
        """
        super().__init__()
        self.output_dir = Path(output_dir)
        self.save_every_n_epochs = save_every_n_epochs
        self.num_samples = num_samples
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule
    ):
        """
        Save sample predictions at validation epoch end.
        
        Args:
            trainer: PyTorch Lightning trainer
            pl_module: Lightning module being trained
        """
        if trainer.current_epoch % self.save_every_n_epochs != 0:
            return
        
        epoch_dir = self.output_dir / f"epoch_{trainer.current_epoch:03d}"
        epoch_dir.mkdir(exist_ok=True)
        
        print(f"\nSaved predictions to {epoch_dir}")


def create_callbacks(config: Dict[str, Any]) -> list:
    """
    Factory function to create all callbacks from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of configured callbacks
    """
    callbacks = []
    
    checkpoint_config = config.get('logging', {}).get('checkpoint', {})
    checkpoints_dir = config.get('paths', {}).get('checkpoints_dir', 'outputs/checkpoints')
    
    checkpoint_callback = CustomModelCheckpoint(
        dirpath=checkpoints_dir,
        monitor=checkpoint_config.get('monitor', 'val_iou_epoch'),
        mode=checkpoint_config.get('mode', 'max'),
        save_top_k=checkpoint_config.get('save_top_k', 3),
        save_last=checkpoint_config.get('save_last', True),
        filename='epoch={epoch:02d}-val_iou={val_iou_epoch:.4f}',
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    early_stopping_config = config.get('training', {}).get('early_stopping', {})
    if early_stopping_config.get('enabled', True):
        early_stopping_callback = CustomEarlyStopping(
            monitor=early_stopping_config.get('monitor', 'val_iou_epoch'),
            patience=early_stopping_config.get('patience', 15),
            mode=early_stopping_config.get('mode', 'max'),
            min_delta=early_stopping_config.get('min_delta', 0.001),
            verbose=True
        )
        callbacks.append(early_stopping_callback)
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    metric_logger = MetricLogger(log_every_n_epochs=1)
    callbacks.append(metric_logger)
    
    gradient_logger = GradientLoggingCallback(log_every_n_steps=100)
    callbacks.append(gradient_logger)
    
    memory_logger = MemoryLoggingCallback(log_every_n_epochs=5)
    callbacks.append(memory_logger)
    
    predictions_dir = config.get('paths', {}).get('predictions_dir', 'outputs/predictions')
    save_predictions_callback = SavePredictionsCallback(
        output_dir=predictions_dir,
        save_every_n_epochs=10,
        num_samples=4
    )
    callbacks.append(save_predictions_callback)
    
    return callbacks