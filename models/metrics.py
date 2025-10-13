"""
Evaluation metrics.
Implements IoU, Dice, Pixel Accuracy, Precision, Recall, and F1 Score.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np


class SegmentationMetrics:
    """
    Comprehensive metrics for semantic/instance segmentation evaluation.
    Computes IoU, Dice, Pixel Accuracy, Precision, Recall, and F1.
    """
    
    def __init__(
        self,
        num_classes: int,
        ignore_index: int = -100,
        reduction: str = 'mean'
    ):
        """
        Initialize segmentation metrics.
        
        Args:
            num_classes: Number of segmentation classes
            ignore_index: Index to ignore in metric calculation
            reduction: Reduction method ('mean', 'none')
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reduction = reduction
        
        self.reset()
    
    def reset(self):
        """Reset all metric accumulators."""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
    
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ):
        """
        Update metrics with new batch of predictions.
        
        Args:
            predictions: Model predictions (B, C, H, W) logits or (B, H, W) indices
            targets: Ground truth masks (B, H, W) class indices
        """
        if predictions.dim() == 4:
            predictions = torch.argmax(predictions, dim=1)
        
        predictions = predictions.cpu().numpy().flatten()
        targets = targets.cpu().numpy().flatten()
        
        if self.ignore_index >= 0:
            mask = targets != self.ignore_index
            predictions = predictions[mask]
            targets = targets[mask]
        
        predictions = np.clip(predictions, 0, self.num_classes - 1)
        targets = np.clip(targets, 0, self.num_classes - 1)
        
        for t, p in zip(targets, predictions):
            self.confusion_matrix[t, p] += 1
    
    def compute_iou(self) -> Dict[str, float]:
        """
        Compute Intersection over Union (IoU) / Jaccard Index.
        
        Returns:
            Dictionary with per-class and mean IoU
        """
        intersection = np.diag(self.confusion_matrix)
        union = (
            self.confusion_matrix.sum(axis=1) + 
            self.confusion_matrix.sum(axis=0) - 
            intersection
        )
        
        iou_per_class = intersection / (union + 1e-10)
        
        valid_classes = union > 0
        mean_iou = iou_per_class[valid_classes].mean()
        
        return {
            'iou_per_class': iou_per_class,
            'mean_iou': mean_iou
        }
    
    def compute_dice(self) -> Dict[str, float]:
        """
        Compute Dice Coefficient / F1 Score.
        
        Returns:
            Dictionary with per-class and mean Dice
        """
        intersection = np.diag(self.confusion_matrix)
        cardinality = (
            self.confusion_matrix.sum(axis=1) + 
            self.confusion_matrix.sum(axis=0)
        )
        
        dice_per_class = (2.0 * intersection) / (cardinality + 1e-10)
        
        valid_classes = cardinality > 0
        mean_dice = dice_per_class[valid_classes].mean()
        
        return {
            'dice_per_class': dice_per_class,
            'mean_dice': mean_dice
        }
    
    def compute_pixel_accuracy(self) -> float:
        """
        Compute pixel-wise accuracy.
        
        Returns:
            Overall pixel accuracy
        """
        correct = np.diag(self.confusion_matrix).sum()
        total = self.confusion_matrix.sum()
        
        return correct / (total + 1e-10)
    
    def compute_precision_recall(self) -> Dict[str, np.ndarray]:
        """
        Compute precision and recall per class.
        
        Returns:
            Dictionary with per-class precision and recall
        """
        true_positives = np.diag(self.confusion_matrix)
        
        predicted_positives = self.confusion_matrix.sum(axis=0)
        actual_positives = self.confusion_matrix.sum(axis=1)
        
        precision = true_positives / (predicted_positives + 1e-10)
        recall = true_positives / (actual_positives + 1e-10)
        
        return {
            'precision_per_class': precision,
            'recall_per_class': recall,
            'mean_precision': precision.mean(),
            'mean_recall': recall.mean()
        }
    
    def compute_f1_score(self) -> Dict[str, np.ndarray]:
        """
        Compute F1 score per class.
        
        Returns:
            Dictionary with per-class and mean F1 score
        """
        pr_metrics = self.compute_precision_recall()
        precision = pr_metrics['precision_per_class']
        recall = pr_metrics['recall_per_class']
        
        f1_per_class = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        valid_classes = (precision + recall) > 0
        mean_f1 = f1_per_class[valid_classes].mean()
        
        return {
            'f1_per_class': f1_per_class,
            'mean_f1': mean_f1
        }
    
    def compute_all(self) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Returns:
            Dictionary with all computed metrics
        """
        iou_metrics = self.compute_iou()
        dice_metrics = self.compute_dice()
        pr_metrics = self.compute_precision_recall()
        f1_metrics = self.compute_f1_score()
        pixel_acc = self.compute_pixel_accuracy()
        
        return {
            'iou': iou_metrics['mean_iou'],
            'dice': dice_metrics['mean_dice'],
            'pixel_accuracy': pixel_acc,
            'precision': pr_metrics['mean_precision'],
            'recall': pr_metrics['mean_recall'],
            'f1': f1_metrics['mean_f1']
        }
    
    def get_confusion_matrix(self) -> np.ndarray:
        """
        Get the confusion matrix.
        
        Returns:
            Confusion matrix (num_classes, num_classes)
        """
        return self.confusion_matrix


def compute_iou(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = -100,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Compute IoU metric for a single batch.
    
    Args:
        predictions: Model predictions (B, C, H, W) or (B, H, W)
        targets: Ground truth masks (B, H, W)
        num_classes: Number of classes
        ignore_index: Index to ignore
        reduction: Reduction method ('mean', 'none')
        
    Returns:
        IoU score(s)
    """
    if predictions.dim() == 4:
        predictions = torch.argmax(predictions, dim=1)
    
    if ignore_index >= 0:
        mask = targets != ignore_index
        predictions = predictions * mask
        targets = targets * mask
    
    iou_per_class = []
    
    for class_id in range(num_classes):
        pred_mask = (predictions == class_id)
        target_mask = (targets == class_id)
        
        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()
        
        if union > 0:
            iou = intersection / union
            iou_per_class.append(iou)
    
    if len(iou_per_class) == 0:
        return torch.tensor(0.0, device=predictions.device)
    
    iou_tensor = torch.stack(iou_per_class)
    
    if reduction == 'mean':
        return iou_tensor.mean()
    else:
        return iou_tensor


def compute_dice(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = -100,
    smooth: float = 1.0,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Compute Dice score for a single batch.
    
    Args:
        predictions: Model predictions (B, C, H, W) or (B, H, W)
        targets: Ground truth masks (B, H, W)
        num_classes: Number of classes
        ignore_index: Index to ignore
        smooth: Smoothing factor
        reduction: Reduction method ('mean', 'none')
        
    Returns:
        Dice score(s)
    """
    if predictions.dim() == 4:
        predictions = torch.argmax(predictions, dim=1)
    
    if ignore_index >= 0:
        mask = targets != ignore_index
        predictions = predictions * mask
        targets = targets * mask
    
    dice_per_class = []
    
    for class_id in range(num_classes):
        pred_mask = (predictions == class_id).float()
        target_mask = (targets == class_id).float()
        
        intersection = (pred_mask * target_mask).sum()
        cardinality = pred_mask.sum() + target_mask.sum()
        
        if cardinality > 0:
            dice = (2.0 * intersection + smooth) / (cardinality + smooth)
            dice_per_class.append(dice)
    
    if len(dice_per_class) == 0:
        return torch.tensor(0.0, device=predictions.device)
    
    dice_tensor = torch.stack(dice_per_class)
    
    if reduction == 'mean':
        return dice_tensor.mean()
    else:
        return dice_tensor


def compute_pixel_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100
) -> torch.Tensor:
    """
    Compute pixel-wise accuracy.
    
    Args:
        predictions: Model predictions (B, C, H, W) or (B, H, W)
        targets: Ground truth masks (B, H, W)
        ignore_index: Index to ignore
        
    Returns:
        Pixel accuracy
    """
    if predictions.dim() == 4:
        predictions = torch.argmax(predictions, dim=1)
    
    if ignore_index >= 0:
        mask = targets != ignore_index
        predictions = predictions[mask]
        targets = targets[mask]
    
    correct = (predictions == targets).sum().float()
    total = targets.numel()
    
    return correct / (total + 1e-10)


class InstanceSegmentationMetrics:
    """
    Metrics specifically for instance segmentation.
    Evaluates detection and segmentation quality at instance level.
    """
    
    def __init__(
        self,
        iou_threshold: float = 0.5,
        ignore_background: bool = True
    ):
        """
        Initialize instance segmentation metrics.
        
        Args:
            iou_threshold: IoU threshold for matching predictions to ground truth
            ignore_background: Whether to ignore background class (ID 0)
        """
        self.iou_threshold = iou_threshold
        self.ignore_background = ignore_background
        
        self.reset()
    
    def reset(self):
        """Reset metric accumulators."""
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.total_iou = 0.0
        self.num_matches = 0
    
    def compute_instance_iou(
        self,
        pred_mask: torch.Tensor,
        target_mask: torch.Tensor,
        pred_id: int,
        target_id: int
    ) -> float:
        """
        Compute IoU between two instance masks.
        
        Args:
            pred_mask: Predicted instance mask
            target_mask: Ground truth instance mask
            pred_id: Predicted instance ID
            target_id: Target instance ID
            
        Returns:
            IoU score
        """
        pred_instance = (pred_mask == pred_id)
        target_instance = (target_mask == target_id)
        
        intersection = (pred_instance & target_instance).sum().float()
        union = (pred_instance | target_instance).sum().float()
        
        if union == 0:
            return 0.0
        
        return (intersection / union).item()
    
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ):
        """
        Update metrics with batch predictions.
        
        Args:
            predictions: Predicted instance masks (B, H, W)
            targets: Ground truth instance masks (B, H, W)
        """
        batch_size = predictions.shape[0]
        
        for b in range(batch_size):
            pred = predictions[b]
            target = targets[b]
            
            pred_ids = torch.unique(pred)
            target_ids = torch.unique(target)
            
            if self.ignore_background:
                pred_ids = pred_ids[pred_ids != 0]
                target_ids = target_ids[target_ids != 0]
            
            matched_targets = set()
            
            for pred_id in pred_ids:
                best_iou = 0.0
                best_target_id = None
                
                for target_id in target_ids:
                    if target_id.item() in matched_targets:
                        continue
                    
                    iou = self.compute_instance_iou(pred, target, pred_id.item(), target_id.item())
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_target_id = target_id.item()
                
                if best_iou >= self.iou_threshold and best_target_id is not None:
                    self.true_positives += 1
                    self.total_iou += best_iou
                    self.num_matches += 1
                    matched_targets.add(best_target_id)
                else:
                    self.false_positives += 1
            
            self.false_negatives += len(target_ids) - len(matched_targets)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute instance-level metrics.
        
        Returns:
            Dictionary with precision, recall, F1, and mean IoU
        """
        precision = self.true_positives / (self.true_positives + self.false_positives + 1e-10)
        recall = self.true_positives / (self.true_positives + self.false_negatives + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        mean_iou = self.total_iou / (self.num_matches + 1e-10)
        
        return {
            'instance_precision': precision,
            'instance_recall': recall,
            'instance_f1': f1,
            'instance_mean_iou': mean_iou
        }


def create_metrics(config: dict) -> SegmentationMetrics:
    """
    Factory function to create metrics from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured SegmentationMetrics instance
    """
    num_classes = config.get('model', {}).get('unet', {}).get('num_classes', 64)
    
    metrics = SegmentationMetrics(
        num_classes=num_classes,
        ignore_index=-100,
        reduction='mean'
    )
    
    return metrics