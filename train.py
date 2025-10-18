"""
Main training script for handwritten character segmentation.
Orchestrates dataset loading, model training, and experiment tracking.
"""

import os
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from config import load_config
from data import create_datamodule
from training import CharacterSegmentationModule, create_callbacks
from models import count_parameters

warnings.filterwarnings('ignore', category=UserWarning)


def setup_wandb_logger(config: dict, run_name: Optional[str] = None) -> Optional[WandbLogger]:
    """
    Setup Weights & Biases logger.
    
    Args:
        config: Configuration dictionary
        run_name: Custom run name (optional)
        
    Returns:
        WandbLogger instance or None if disabled
    """
    wandb_config = config.get('logging', {}).get('wandb', {})
    
    if not wandb_config.get('enabled', True):
        print("Weights & Biases logging disabled")
        return None
    
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"char_seg_{timestamp}"
    
    logger = WandbLogger(
        project=wandb_config.get('project', 'handwritten-char-segmentation'),
        entity=wandb_config.get('entity', None),
        name=run_name,
        log_model=wandb_config.get('log_model', True),
        save_dir=config.get('paths', {}).get('logs_dir', 'outputs/logs')
    )
    
    try:
        import wandb
        if wandb.run is not None:
            wandb.config.update(config, allow_val_change=True)
    except Exception as e:
        print(f"Warning: Could not update wandb config: {e}")
    
    return logger


def print_training_info(config: dict, model: pl.LightningModule, datamodule: pl.LightningDataModule):
    """
    Print detailed training information.
    
    Args:
        config: Configuration dictionary
        model: Lightning module
        datamodule: Data module
    """
    print("\n" + "=" * 80)
    print("HANDWRITTEN CHARACTER SEGMENTATION - TRAINING")
    print("=" * 80)
    
    print("\n📊 MODEL ARCHITECTURE")
    print("-" * 80)
    num_params = count_parameters(model.model)
    print(f"Architecture: U-Net with Attention Gates")
    print(f"Total Parameters: {num_params:,}")
    print(f"Trainable Parameters: {num_params:,}")
    print(f"Number of Classes: {model.num_classes}")
    
    print("\n📁 DATASET INFORMATION")
    print("-" * 80)
    data_dir = config.get('paths', {}).get('synthetic_data_dir')
    print(f"Data Directory: {data_dir}")
    print(f"Batch Size (Train): {datamodule.batch_size}")
    print(f"Batch Size (Val): {datamodule.val_batch_size}")
    print(f"Num Workers: {datamodule.num_workers}")
    
    print("\n⚙️  TRAINING CONFIGURATION")
    print("-" * 80)
    training_config = config.get('training', {})
    print(f"Optimizer: {training_config.get('optimizer', {}).get('name', 'adamw').upper()}")
    print(f"Learning Rate: {training_config.get('optimizer', {}).get('learning_rate', 0.001)}")
    print(f"Weight Decay: {training_config.get('optimizer', {}).get('weight_decay', 0.0001)}")
    print(f"Scheduler: {training_config.get('scheduler', {}).get('name', 'cosine')}")
    print(f"Loss Function: {training_config.get('loss', {}).get('name', 'combined')}")
    print(f"Epochs: {training_config.get('num_epochs', 100)}")
    print(f"Mixed Precision: {training_config.get('use_amp', True)}")
    print(f"Gradient Clipping: {training_config.get('gradient_clip_val', 1.0)}")
    
    print("\n🎯 EARLY STOPPING")
    print("-" * 80)
    es_config = training_config.get('early_stopping', {})
    if es_config.get('enabled', True):
        print(f"Enabled: Yes")
        print(f"Monitor: {es_config.get('monitor', 'val_iou')}")
        print(f"Patience: {es_config.get('patience', 15)} epochs")
        print(f"Min Delta: {es_config.get('min_delta', 0.001)}")
    else:
        print(f"Enabled: No")
    
    print("\n💾 CHECKPOINTING")
    print("-" * 80)
    checkpoint_config = config.get('logging', {}).get('checkpoint', {})
    print(f"Monitor: {checkpoint_config.get('monitor', 'val_iou')}")
    print(f"Save Top K: {checkpoint_config.get('save_top_k', 3)}")
    print(f"Save Last: {checkpoint_config.get('save_last', True)}")
    
    print("\n🔧 HARDWARE")
    print("-" * 80)
    hardware_config = config.get('hardware', {})
    device = hardware_config.get('device', 'cuda')
    print(f"Device: {device.upper()}")
    
    if device == 'cuda' and torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"GPUs Available: {gpu_count}")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    print("\n" + "=" * 80 + "\n")


def train(
    config_path: str = 'config/config.yaml',
    resume_from_checkpoint: Optional[str] = None,
    run_name: Optional[str] = None,
    seed: Optional[int] = None
):
    """
    Main training function.
    
    Args:
        config_path: Path to configuration file
        resume_from_checkpoint: Path to checkpoint to resume from
        run_name: Custom run name for W&B
        seed: Random seed (overrides config)
    """
    config = load_config(config_path)
    
    if seed is not None:
        config.set('project.seed', seed)
    
    pl.seed_everything(config.get('project.seed'), workers=True)
    
    datamodule = create_datamodule(config.to_dict())
    datamodule.prepare_data()
    datamodule.setup(stage='fit')
    
    model = CharacterSegmentationModule(config.to_dict())
    
    logger = setup_wandb_logger(config.to_dict(), run_name)
    
    callbacks = create_callbacks(config.to_dict())
    
    print_training_info(config.to_dict(), model, datamodule)
    
    hardware_config = config.get('hardware', {})
    accelerator = 'gpu' if hardware_config.get('device', 'cuda') == 'cuda' and torch.cuda.is_available() else 'cpu'
    
    devices = 'auto'
    strategy = 'auto'
    
    if hardware_config.get('distributed', False) and torch.cuda.device_count() > 1:
        devices = hardware_config.get('gpu_ids', list(range(torch.cuda.device_count())))
        strategy = DDPStrategy(find_unused_parameters=False)
        print(f"Using Distributed Data Parallel with {len(devices)} GPUs")
    
    precision = hardware_config.get('precision', 16)
    if accelerator == 'cpu':
        precision = 32
    
    trainer = pl.Trainer(
        max_epochs=config.get('training.num_epochs'),
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=precision,
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=config.get('training.gradient_clip_val'),
        log_every_n_steps=50,
        check_val_every_n_epoch=config.get('validation.check_val_every_n_epoch'),
        deterministic=False,
        benchmark=True,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    print("🚀 Starting training...\n")
    
    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=resume_from_checkpoint
    )
    
    print("\n✅ Training completed!")
    
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_model_score = trainer.checkpoint_callback.best_model_score
    
    print("\n" + "=" * 80)
    print("TRAINING RESULTS")
    print("=" * 80)
    print(f"Best Model: {best_model_path}")
    print(f"Best Val IoU: {best_model_score:.4f}")
    print("=" * 80 + "\n")
    
    if logger is not None:
        print(f"📊 View results at: {logger.experiment.url}")
    
    return trainer, model


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train handwritten character segmentation model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py
  python train.py --config config/custom_config.yaml
  python train.py --resume outputs/checkpoints/best_model.ckpt
  python train.py --run_name experiment_v2 --seed 123
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )
    
    parser.add_argument(
        '--run_name',
        type=str,
        default=None,
        help='Custom run name for Weights & Biases'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed (overrides config)'
    )
    
    parser.add_argument(
        '--fast_dev_run',
        action='store_true',
        help='Run quick test with 1 batch (for debugging)'
    )
    
    args = parser.parse_args()
    
    if args.fast_dev_run:
        print("⚡ Fast dev run mode - testing with 1 batch")
    
    try:
        train(
            config_path=args.config,
            resume_from_checkpoint=args.resume,
            run_name=args.run_name,
            seed=args.seed
        )
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()