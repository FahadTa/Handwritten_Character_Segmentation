"""
Loss functions for character instance segmentation.
Implements Dice, Cross-Entropy, Combined, and Focal losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks.
    Effective for handling class imbalance in segmentation.
    """
    
    def __init__(
        self,
        smooth: float = 1.0,
        ignore_index: int = -100,
        reduction: str = 'mean'
    ):
        """
        Initialize Dice Loss.
        
        Args:
            smooth: Smoothing factor to avoid division by zero
            ignore_index: Index to ignore in loss calculation
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.reduction = reduction
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Dice loss.
        
        Args:
            predictions: Model predictions (B, C, H, W) - logits
            targets: Ground truth masks (B, H, W) - class indices
            
        Returns:
            Dice loss value
        """
        num_classes = predictions.shape[1]
        
        predictions = F.softmax(predictions, dim=1)
        
        targets = targets.long().clamp(0, num_classes - 1)
        
        targets_one_hot = F.one_hot(
            targets, 
            num_classes=num_classes
        )
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        
        if self.ignore_index >= 0:
            mask = (targets != self.ignore_index).float().unsqueeze(1)
            predictions = predictions * mask
            targets_one_hot = targets_one_hot * mask
        
        predictions_flat = predictions.contiguous().view(-1)
        targets_flat = targets_one_hot.contiguous().view(-1)
        
        intersection = (predictions_flat * targets_flat).sum()
        
        dice_score = (2.0 * intersection + self.smooth) / (
            predictions_flat.sum() + targets_flat.sum() + self.smooth
        )
        
        dice_loss = 1.0 - dice_score
        
        if self.reduction == 'mean':
            return dice_loss
        elif self.reduction == 'sum':
            return dice_loss * predictions.shape[0]
        else:
            return dice_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Focuses training on hard examples.
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        ignore_index: int = -100,
        reduction: str = 'mean'
    ):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for class balance
            gamma: Focusing parameter for hard examples
            ignore_index: Index to ignore in loss calculation
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Focal loss.
        
        Args:
            predictions: Model predictions (B, C, H, W) - logits
            targets: Ground truth masks (B, H, W) - class indices
            
        Returns:
            Focal loss value
        """
        num_classes = predictions.shape[1]
        targets = targets.long().clamp(0, num_classes - 1)
        
        ce_loss = F.cross_entropy(
            predictions,
            targets,
            ignore_index=self.ignore_index,
            reduction='none'
        )
        
        p = torch.exp(-ce_loss)
        
        focal_loss = self.alpha * (1 - p) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedLoss(nn.Module):
    """
    Combined loss: weighted sum of Dice loss and Cross-Entropy loss.
    Leverages benefits of both loss functions for robust training.
    """
    
    def __init__(
        self,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        smooth: float = 1.0
    ):
        """
        Initialize Combined Loss.
        
        Args:
            dice_weight: Weight for Dice loss component
            ce_weight: Weight for Cross-Entropy loss component
            class_weights: Optional class weights for CE loss
            ignore_index: Index to ignore in loss calculation
            smooth: Smoothing factor for Dice loss
        """
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.ignore_index = ignore_index
        
        self.dice_loss = DiceLoss(
            smooth=smooth,
            ignore_index=ignore_index,
            reduction='mean'
        )
        
        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index,
            reduction='mean'
        )
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            predictions: Model predictions (B, C, H, W) - logits
            targets: Ground truth masks (B, H, W) - class indices
            
        Returns:
            Combined loss value
        """
        num_classes = predictions.shape[1]
        targets = targets.long().clamp(0, num_classes - 1)
        
        dice = self.dice_loss(predictions, targets)
        ce = self.ce_loss(predictions, targets)
        
        combined = self.dice_weight * dice + self.ce_weight * ce
        
        return combined


class TverskyLoss(nn.Module):
    """
    Tversky Loss - generalization of Dice loss.
    Allows tuning the trade-off between false positives and false negatives.
    """
    
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        smooth: float = 1.0,
        ignore_index: int = -100
    ):
        """
        Initialize Tversky Loss.
        
        Args:
            alpha: Weight for false positives
            beta: Weight for false negatives
            smooth: Smoothing factor
            ignore_index: Index to ignore in loss calculation
        """
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.ignore_index = ignore_index
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Tversky loss.
        
        Args:
            predictions: Model predictions (B, C, H, W) - logits
            targets: Ground truth masks (B, H, W) - class indices
            
        Returns:
            Tversky loss value
        """
        num_classes = predictions.shape[1]
        targets = targets.long().clamp(0, num_classes - 1)
        
        predictions = F.softmax(predictions, dim=1)
        
        targets_one_hot = F.one_hot(
            targets,
            num_classes=num_classes
        )
        
        targets_one_hot = F.one_hot(
            targets.long().clamp(0, num_classes - 1),
            num_classes=num_classes
        )
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        
        if self.ignore_index >= 0:
            mask = (targets != self.ignore_index).float().unsqueeze(1)
            predictions = predictions * mask
            targets_one_hot = targets_one_hot * mask
        
        true_pos = (predictions * targets_one_hot).sum()
        false_pos = (predictions * (1 - targets_one_hot)).sum()
        false_neg = ((1 - predictions) * targets_one_hot).sum()
        
        tversky_index = (true_pos + self.smooth) / (
            true_pos + self.alpha * false_pos + self.beta * false_neg + self.smooth
        )
        
        return 1.0 - tversky_index


def get_loss_function(config: dict, device: torch.device) -> nn.Module:
    """
    Factory function to create loss function from configuration.
    
    Args:
        config: Configuration dictionary containing loss settings
        device: Device to place loss function on
        
    Returns:
        Configured loss function
    """
    loss_config = config.get('training', {}).get('loss', {})
    loss_name = loss_config.get('name', 'combined').lower()
    
    class_weights = loss_config.get('class_weights', None)
    if class_weights is not None:
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    if loss_name == 'dice':
        return DiceLoss(
            smooth=1.0,
            ignore_index=-100,
            reduction='mean'
        )
    
    elif loss_name == 'ce' or loss_name == 'cross_entropy':
        return nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=-100,
            reduction='mean'
        )
    
    elif loss_name == 'combined':
        return CombinedLoss(
            dice_weight=loss_config.get('dice_weight', 0.5),
            ce_weight=loss_config.get('ce_weight', 0.5),
            class_weights=class_weights,
            ignore_index=-100,
            smooth=1.0
        )
    
    elif loss_name == 'focal':
        return FocalLoss(
            alpha=loss_config.get('focal_alpha', 0.25),
            gamma=loss_config.get('focal_gamma', 2.0),
            ignore_index=-100,
            reduction='mean'
        )
    
    elif loss_name == 'tversky':
        return TverskyLoss(
            alpha=0.5,
            beta=0.5,
            smooth=1.0,
            ignore_index=-100
        )
    
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")