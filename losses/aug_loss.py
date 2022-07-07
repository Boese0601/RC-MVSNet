import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def random_image_mask(img, filter_size):
    '''

    :param img: [B x 3 x H x W]
    :param crop_size:
    :return:
    '''
    fh, fw = filter_size
    _, _, h, w = img.size()

    if fh == h and fw == w:
        return img, None

    x = np.random.randint(0, w - fw)
    y = np.random.randint(0, h - fh)
    filter_mask = torch.ones_like(img)    # B x 3 x H x W
    filter_mask[:, :, y:y+fh, x:x+fw] = 0.0    # B x 3 x H x W
    img = img * filter_mask    # B x 3 x H x W
    return img, filter_mask


class AugLossMultiStage(nn.Module):
    def __init__(self):
        super(AugLossMultiStage, self).__init__()

    def forward(self, inputs, pseudo_depth, mask_ms, filter_mask, **kwargs):
        depth_loss_weights = kwargs.get("dlossw", None)

        total_loss = torch.tensor(0.0, dtype=torch.float32, device=pseudo_depth.device, requires_grad=False)

        scalar_outputs = {}
        for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
            stage_idx = int(stage_key.replace("stage", "")) - 1

            depth_est = stage_inputs["depth"]

            pseudo_gt = pseudo_depth.unsqueeze(dim=1)
            if stage_idx == 0:
                pseudo_gt_t = F.interpolate(pseudo_gt, scale_factor=(0.25, 0.25),recompute_scale_factor=True)
                filter_mask_t = F.interpolate(filter_mask, scale_factor=(0.25, 0.25),recompute_scale_factor=True)
            elif stage_idx == 1:
                pseudo_gt_t = F.interpolate(pseudo_gt, scale_factor=(0.5, 0.5),recompute_scale_factor=True)
                filter_mask_t = F.interpolate(filter_mask, scale_factor=(0.5, 0.5),recompute_scale_factor=True)
            else:
                pseudo_gt_t = pseudo_gt
                filter_mask_t = filter_mask
            filter_mask_t = filter_mask_t[:, 0, :, :]
            pseudo_gt_t = pseudo_gt_t.squeeze(dim=1)

            mask = filter_mask_t > 0.5
            # print('depth_est: {} pseudo_gt_t: {} mask: {}'.format(depth_est.shape, pseudo_gt_t.shape, mask.shape))
            depth_loss = F.smooth_l1_loss(depth_est[mask], pseudo_gt_t[mask], reduction='mean')

            if depth_loss_weights is not None:
                total_loss += depth_loss_weights[stage_idx] * depth_loss
            else:
                total_loss += 1.0 * depth_loss

            scalar_outputs["aug_loss_stage{}".format(stage_idx + 1)] = depth_loss

        return total_loss, scalar_outputs
