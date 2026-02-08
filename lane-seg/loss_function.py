import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=2.0, smooth=1e-6):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha   # Weight for False Negatives (Recall)
        self.beta = beta     # Weight for False Positives (Precision)
        self.gamma = gamma   # Focal parameter (higher = more focus on hard cases)
        self.smooth = smooth

    def forward(self, pred, target):
        # Apply sigmoid for binary segmentation
        pred = torch.sigmoid(pred)
        
        # Flatten tensors for calculation
        pred = pred.view(-1)
        target = target.view(-1)
        
        # Calculate components
        tp = (pred * target).sum()
        fp = ((1 - target) * pred).sum()
        fn = (target * (1 - pred)).sum()
        
        tversky_index = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        
        # Focal Tversky Loss: (1 - TI)^gamma
        # Gamma > 1 pushes the model to focus on pixels it is struggling to classify
        loss = torch.pow((1 - tversky_index), self.gamma)
        
        return loss

class LaneSegmentationLoss(nn.Module):
    def __init__(self, bce_weight=0.2, tversky_alpha=0.3, gamma=2.0):
        super(LaneSegmentationLoss, self).__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.focal_tversky = FocalTverskyLoss(alpha=tversky_alpha, beta=1-tversky_alpha, gamma=gamma)

    def forward(self, pred, target):
        # BCE provides smooth gradient flow for general structure
        bce_loss = self.bce(pred, target)
        
        # Focal Tversky handles the thin, sparse lane pixels
        ft_loss = self.focal_tversky(pred, target)
        
        # Weighted combination
        return (self.bce_weight * bce_loss) + ((1 - self.bce_weight) * ft_loss)