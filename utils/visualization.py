"""
Visualization utilities for handwritten character segmentation.
Creates publication-quality visualizations of predictions and results.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import cv2
import torch


class SegmentationVisualizer:
    """
    Comprehensive visualization toolkit for character segmentation results.
    """
    
    def __init__(self, save_dir: Optional[str] = None):
        """
        Initialize visualizer.
        
        Args:
            save_dir: Directory to save visualizations
        """
        self.save_dir = Path(save_dir) if save_dir else Path("outputs/visualizations")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        plt.style.use('seaborn-v0_8-darkgrid')
        self.cmap = plt.cm.get_cmap('tab20')
    
    def visualize_prediction(
        self,
        image: np.ndarray,
        ground_truth: np.ndarray,
        prediction: np.ndarray,
        title: str = "Segmentation Result",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize image with ground truth and prediction masks.
        
        Args:
            image: Input image (H, W, 3) in range [0, 255]
            ground_truth: Ground truth mask (H, W)
            prediction: Predicted mask (H, W)
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        axes[0].imshow(image)
        axes[0].set_title("Input Image", fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        gt_colored = self._colorize_mask(ground_truth)
        axes[1].imshow(gt_colored)
        axes[1].set_title("Ground Truth", fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        pred_colored = self._colorize_mask(prediction)
        axes[2].imshow(pred_colored)
        axes[2].set_title("Prediction", fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        overlay = self._create_overlay(image, prediction)
        axes[3].imshow(overlay)
        axes[3].set_title("Overlay", fontsize=14, fontweight='bold')
        axes[3].axis('off')
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        
        return fig
    
    def visualize_batch(
        self,
        images: torch.Tensor,
        ground_truths: torch.Tensor,
        predictions: torch.Tensor,
        num_samples: int = 4,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize multiple samples in a grid.
        
        Args:
            images: Batch of images (B, 3, H, W)
            ground_truths: Batch of GT masks (B, H, W)
            predictions: Batch of predictions (B, H, W)
            num_samples: Number of samples to visualize
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        num_samples = min(num_samples, images.shape[0])
        
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
        
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_samples):
            img = self._denormalize_image(images[i])
            gt = ground_truths[i].cpu().numpy()
            pred = predictions[i].cpu().numpy() if torch.is_tensor(predictions[i]) else predictions[i]
            
            axes[i, 0].imshow(img)
            axes[i, 0].set_title("Input", fontsize=12, fontweight='bold')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(self._colorize_mask(gt))
            axes[i, 1].set_title("Ground Truth", fontsize=12, fontweight='bold')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(self._colorize_mask(pred))
            axes[i, 2].set_title("Prediction", fontsize=12, fontweight='bold')
            axes[i, 2].axis('off')
        
        plt.suptitle("Batch Predictions", fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved batch visualization to {save_path}")
        
        return fig
    
    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot training and validation metrics over epochs.
        
        Args:
            history: Dictionary with metric histories
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        if 'train_loss' in history and 'val_loss' in history:
            axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
            axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
            axes[0, 0].set_xlabel('Epoch', fontsize=12)
            axes[0, 0].set_ylabel('Loss', fontsize=12)
            axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
            axes[0, 0].legend(fontsize=11)
            axes[0, 0].grid(True, alpha=0.3)
        
        if 'train_iou' in history and 'val_iou' in history:
            axes[0, 1].plot(history['train_iou'], label='Train IoU', linewidth=2)
            axes[0, 1].plot(history['val_iou'], label='Val IoU', linewidth=2)
            axes[0, 1].set_xlabel('Epoch', fontsize=12)
            axes[0, 1].set_ylabel('IoU', fontsize=12)
            axes[0, 1].set_title('Intersection over Union', fontsize=14, fontweight='bold')
            axes[0, 1].legend(fontsize=11)
            axes[0, 1].grid(True, alpha=0.3)
        
        if 'train_dice' in history and 'val_dice' in history:
            axes[1, 0].plot(history['train_dice'], label='Train Dice', linewidth=2)
            axes[1, 0].plot(history['val_dice'], label='Val Dice', linewidth=2)
            axes[1, 0].set_xlabel('Epoch', fontsize=12)
            axes[1, 0].set_ylabel('Dice Score', fontsize=12)
            axes[1, 0].set_title('Dice Coefficient', fontsize=14, fontweight='bold')
            axes[1, 0].legend(fontsize=11)
            axes[1, 0].grid(True, alpha=0.3)
        
        if 'learning_rate' in history:
            axes[1, 1].plot(history['learning_rate'], linewidth=2, color='green')
            axes[1, 1].set_xlabel('Epoch', fontsize=12)
            axes[1, 1].set_ylabel('Learning Rate', fontsize=12)
            axes[1, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved training history to {save_path}")
        
        return fig
    
    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        class_names: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot confusion matrix for character classification.
        
        Args:
            confusion_matrix: Confusion matrix (C, C)
            class_names: List of class names
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        im = ax.imshow(confusion_matrix, cmap='Blues', aspect='auto')
        
        num_classes = confusion_matrix.shape[0]
        if class_names is None:
            class_names = [f"Class {i}" for i in range(num_classes)]
        
        ax.set_xticks(np.arange(num_classes))
        ax.set_yticks(np.arange(num_classes))
        ax.set_xticklabels(class_names, rotation=90, fontsize=8)
        ax.set_yticklabels(class_names, fontsize=8)
        
        ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
        ax.set_ylabel('Ground Truth', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved confusion matrix to {save_path}")
        
        return fig
    
    def plot_metrics_comparison(
        self,
        metrics: Dict[str, float],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot bar chart comparing different metrics.
        
        Args:
            metrics: Dictionary of metric names and values
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(metric_names)))
        bars = ax.bar(metric_names, metric_values, color=colors, edgecolor='black', linewidth=1.5)
        
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.4f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Evaluation Metrics Comparison', fontsize=14, fontweight='bold')
        ax.set_ylim(0, max(metric_values) * 1.15)
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved metrics comparison to {save_path}")
        
        return fig
    
    def _colorize_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Convert instance mask to RGB visualization.
        
        Args:
            mask: Instance mask (H, W)
            
        Returns:
            RGB image (H, W, 3)
        """
        colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        
        unique_ids = np.unique(mask)
        
        for idx, instance_id in enumerate(unique_ids):
            if instance_id == 0:
                continue
            
            color = self.cmap(idx % 20)[:3]
            color_uint8 = (np.array(color) * 255).astype(np.uint8)
            
            colored_mask[mask == instance_id] = color_uint8
        
        return colored_mask
    
    def _create_overlay(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Create overlay of mask on image.
        
        Args:
            image: Input image (H, W, 3)
            mask: Segmentation mask (H, W)
            alpha: Transparency factor
            
        Returns:
            Overlayed image
        """
        colored_mask = self._colorize_mask(mask)
        
        overlay = cv2.addWeighted(
            image.astype(np.uint8),
            1 - alpha,
            colored_mask,
            alpha,
            0
        )
        
        return overlay
    
    def _denormalize_image(self, image: torch.Tensor) -> np.ndarray:
        """
        Denormalize image tensor for visualization.
        
        Args:
            image: Image tensor (3, H, W) normalized
            
        Returns:
            Denormalized image (H, W, 3) in [0, 255]
        """
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        
        image = image.cpu().numpy()
        image = (image * std + mean) * 255
        image = np.clip(image, 0, 255).astype(np.uint8)
        image = np.transpose(image, (1, 2, 0))
        
        return image


def save_predictions_grid(
    images: List[np.ndarray],
    predictions: List[np.ndarray],
    ground_truths: Optional[List[np.ndarray]] = None,
    save_path: str = "predictions_grid.png",
    max_images: int = 16
):
    """
    Save grid of predictions for quick inspection.
    
    Args:
        images: List of input images
        predictions: List of prediction masks
        ground_truths: Optional list of ground truth masks
        save_path: Path to save grid
        max_images: Maximum number of images to show
    """
    num_images = min(len(images), max_images)
    cols = 4
    rows = (num_images + cols - 1) // cols
    
    num_cols = 3 if ground_truths else 2
    fig, axes = plt.subplots(rows, cols * num_cols, figsize=(5 * cols, 5 * rows))
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_images):
        row = i // cols
        col = i % cols
        
        base_col = col * num_cols
        
        axes[row, base_col].imshow(images[i])
        axes[row, base_col].axis('off')
        axes[row, base_col].set_title(f"Image {i+1}", fontsize=10)
        
        if ground_truths:
            axes[row, base_col + 1].imshow(ground_truths[i], cmap='tab20')
            axes[row, base_col + 1].axis('off')
            axes[row, base_col + 1].set_title(f"GT {i+1}", fontsize=10)
            
            axes[row, base_col + 2].imshow(predictions[i], cmap='tab20')
            axes[row, base_col + 2].axis('off')
            axes[row, base_col + 2].set_title(f"Pred {i+1}", fontsize=10)
        else:
            axes[row, base_col + 1].imshow(predictions[i], cmap='tab20')
            axes[row, base_col + 1].axis('off')
            axes[row, base_col + 1].set_title(f"Pred {i+1}", fontsize=10)
    
    for i in range(num_images, rows * cols):
        row = i // cols
        col = i % cols
        for j in range(num_cols):
            axes[row, col * num_cols + j].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved predictions grid to {save_path}")