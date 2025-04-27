import torch
import torch.nn as nn
import torch.nn.functional as F


class Criterion(nn.Module):
    def __init__(self, device, class_weights=None):
        """
        Combines penalized transitions with class weights for cross-entropy loss.

        Args:
            device (torch.device): Device to perform loss criterion on.
            allowed_transitions (dict): Allowed transitions between classes.
            class_weights (torch.Tensor): Precomputed weights for each class.
            penalty_weight (float): Penalty for invalid transitions.
        """
        super(Criterion, self).__init__()
        self.device = device
        self.class_weights = class_weights


    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Logits of shape (batch_size, num_future_samples, num_classes).
            targets (torch.Tensor): Ground truth of shape (batch_size, num_future_samples).

        Returns:
            torch.Tensor: Combined penalized and weighted loss.
        """
        batch_size, num_future_samples, num_classes = logits.shape
        loss = 0
        # Case 1: No class weights
        if self.class_weights is None:
            for t in range(num_future_samples):
                target_t = targets[:, t] # Shape: (batch_size, 1)
                logits_t = logits[:, t, :]  # Shape: (batch_size, num_classes)
                loss += F.cross_entropy(logits_t, target_t)

        # Case 2:  Class weights
        else:
            for t in range(num_future_samples):
                target_t = targets[:, t]  # Shape: (batch_size,1)
                logits_t = logits[:, t, :]  # Shape: (batch_size, num_classes)
                loss += F.cross_entropy(logits_t, target_t, weight=self.class_weights.to(self.device))

        # Normalize by the number of future samples
        return loss / num_future_samples
