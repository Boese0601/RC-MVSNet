import torch
import torch.nn as nn

class SL1Loss(nn.Module):
    def __init__(self):
        super(SL1Loss, self).__init__()
        self.loss = nn.SmoothL1Loss(reduction='mean')

    def forward(self, depth_pred, depth_gt, mask=None):
        if None == mask:
            mask = depth_gt > 0
        loss = self.loss(depth_pred[mask], depth_gt[mask]) * 2 ** (1 - 2)
        return loss
        