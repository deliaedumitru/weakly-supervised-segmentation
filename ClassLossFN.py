import torch
import numpy as np
from torch import nn

class ClassLossFN(torch.nn.Module):
    
    def __init__(self):
        super(ClassLossFN, self).__init__()
        self.class_loss_fn = nn.BCELoss()
        
    def forward(self, pred, gt, labels):

        sh = pred.size()
        pred = pred.view(sh[0], sh[1], sh[2]*sh[3]).contiguous()
        pred_sorted, _ = pred.sort(dim=2)
        
        weights = np.zeros((sh[0], sh[1], sh[2]*sh[3]))
        for b, class_gt_batch in enumerate(labels):
            for c, class_gt in enumerate(class_gt_batch):
                q_fg = 0 if class_gt.item() == 0 else 0.99
                weights[b, c, :] = np.array([ q_fg ** i for i in range(sh[2]*sh[3] - 1, -1, -1)])
        weights = torch.Tensor(weights)
        if pred_sorted.get_device() >= 0:
            weights = weights.to(pred_sorted.get_device())
        Z_fg = weights.sum(dim=2, keepdim=True)
        pred_normalized = pred_sorted * weights
        pred_normalized = pred_normalized / Z_fg 
        pred_mean = torch.sum(pred_normalized, dim=2)

        return self.class_loss_fn(pred_mean, labels)