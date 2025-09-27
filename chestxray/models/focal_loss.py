import torch
import torch.nn as nn
import torch.nn.functional as F

class MultilabelFocalLoss(torch.nn.Module):
    """
    Focal Loss for multilabel classification
    
    Args:
        alpha: Weighting factor for rare class (default: 1.0)
        gamma: Focusing parameter (default: 2.0)
        pos_weight: Additional positive class weighting (like BCE)
    """
    def __init__(self, alpha=1.0, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
    
    
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        
        # Calculate base BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction='none'
        )
        
        # Calculate focal weight
        # For positive samples: (1 - p)^gamma
        # For negative samples: p^gamma
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        
        alpha = self.alpha
        if alpha.device != logits.device:
            alpha = alpha.to(logits.device)
        # Apply alpha weighting (typically higher for positive class)
        alpha_weight = torch.where(targets == 1, alpha, 1.0)
        
        focal_loss = alpha_weight * focal_weight * bce_loss
        
        return focal_loss.mean()