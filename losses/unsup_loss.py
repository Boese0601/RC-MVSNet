import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.modules import *
from losses.homography import *


class UnSupLoss(nn.Module):
    def __init__(self):
        super(UnSupLoss, self).__init__()
        self.ssim = SSIM()

    def forward(self, imgs, cams, depth, stage_idx):
        # print('imgs: {}'.format(imgs.shape))
        # print('cams: {}'.format(cams.shape))
        # print('depth: {}'.format(depth.shape))

        imgs = torch.unbind(imgs, 1)
        cams = torch.unbind(cams, 1)
        assert len(imgs) == len(cams), "Different number of images and projection matrices"
        img_height, img_width = imgs[0].shape[2], imgs[0].shape[3]
        num_views = len(imgs)

        ref_img = imgs[0]

        if stage_idx == 0:
            ref_img = F.interpolate(ref_img, scale_factor=0.25,recompute_scale_factor=True)
        elif stage_idx == 1:
            ref_img = F.interpolate(ref_img, scale_factor=0.5,recompute_scale_factor=True)
        else:
            pass
        ref_img = ref_img.permute(0, 2, 3, 1)  # [B, C, H, W] --> [B, H, W, C]
        ref_cam = cams[0]
        # print('ref_cam: {}'.format(ref_cam.shape))

        # depth reshape
        # depth = depth.unsqueeze(dim=1)  # [B, 1, H, W]
        # depth = F.interpolate(depth, size=[img_height, img_width])
        # depth = depth.squeeze(dim=1)  # [B, H, W]

        self.reconstr_loss = 0
        self.ssim_loss = 0
        self.smooth_loss = 0

        warped_img_list = []
        mask_list = []
        reprojection_losses = []
        for view in range(1, num_views):
            view_img = imgs[view]
            view_cam = cams[view]
            # print('view_cam: {}'.format(view_cam.shape))
            # view_img = F.interpolate(view_img, scale_factor=0.25, mode='bilinear')
            if stage_idx == 0:
                view_img = F.interpolate(view_img, scale_factor=0.25,recompute_scale_factor=True)
            elif stage_idx == 1:
                view_img = F.interpolate(view_img, scale_factor=0.5,recompute_scale_factor=True)
            else:
                pass
            view_img = view_img.permute(0, 2, 3, 1)  # [B, C, H, W] --> [B, H, W, C]
            # warp view_img to the ref_img using the dmap of the ref_img
            warped_img, mask = inverse_warping(view_img, ref_cam, view_cam, depth)
            warped_img_list.append(warped_img)
            mask_list.append(mask)

            reconstr_loss = compute_reconstr_loss(warped_img, ref_img, mask, simple=False)
            valid_mask = 1 - mask  # replace all 0 values with INF
            reprojection_losses.append(reconstr_loss + 1e4 * valid_mask)

            # SSIM loss##
            if view < 3:
                self.ssim_loss += torch.mean(self.ssim(ref_img, warped_img, mask))

        ##smooth loss##
        self.smooth_loss += depth_smoothness(depth.unsqueeze(dim=-1), ref_img, 1.0)

        # top-k operates along the last dimension, so swap the axes accordingly
        reprojection_volume = torch.stack(reprojection_losses).permute(1, 2, 3, 4, 0)
        # print('reprojection_volume: {}'.format(reprojection_volume.shape))
        # by default, it'll return top-k largest entries, hence sorted=False to get smallest entries
        # top_vals, top_inds = torch.topk(torch.neg(reprojection_volume), k=3, sorted=False)
        top_vals, top_inds = torch.topk(torch.neg(reprojection_volume), k=1, sorted=False)
        top_vals = torch.neg(top_vals)
        # top_mask = top_vals < (1e4 * torch.ones_like(top_vals, device=device))
        top_mask = top_vals < (1e4 * torch.ones_like(top_vals).cuda())
        top_mask = top_mask.float()
        top_vals = torch.mul(top_vals, top_mask)
        # print('top_vals: {}'.format(top_vals.shape))

        self.reconstr_loss = torch.mean(torch.sum(top_vals, dim=-1))
        self.unsup_loss = 12 * self.reconstr_loss + 6 * self.ssim_loss + 0.18 * self.smooth_loss
        # 按照un_mvsnet和M3VSNet的设置
        # self.unsup_loss = (0.8 * self.reconstr_loss + 0.2 * self.ssim_loss + 0.067 * self.smooth_loss) * 15
        return self.unsup_loss

class UnSupLoss_no_smooth(nn.Module):
    def __init__(self):
        super(UnSupLoss_no_smooth, self).__init__()
        self.ssim = SSIM()

    def forward(self, imgs, cams, depth, stage_idx):
        # print('imgs: {}'.format(imgs.shape))
        # print('cams: {}'.format(cams.shape))
        # print('depth: {}'.format(depth.shape))

        imgs = torch.unbind(imgs, 1)
        cams = torch.unbind(cams, 1)
        assert len(imgs) == len(cams), "Different number of images and projection matrices"
        img_height, img_width = imgs[0].shape[2], imgs[0].shape[3]
        num_views = len(imgs)

        ref_img = imgs[0]
        # ref_img = F.interpolate(ref_img, scale_factor=0.25, mode='bilinear')
        # 按照stage进行resize，匹配到每个阶段的分辨率
        # 这里尽量不要使用bilinear，这个会平滑图像的边缘，可能会对自监督损失有影响
        if stage_idx == 0:
            ref_img = F.interpolate(ref_img, scale_factor=0.25,recompute_scale_factor=True)
        elif stage_idx == 1:
            ref_img = F.interpolate(ref_img, scale_factor=0.5,recompute_scale_factor=True)
        else:
            pass
        ref_img = ref_img.permute(0, 2, 3, 1)  # [B, C, H, W] --> [B, H, W, C]
        ref_cam = cams[0]
        # print('ref_cam: {}'.format(ref_cam.shape))

        # depth reshape
        # depth = depth.unsqueeze(dim=1)  # [B, 1, H, W]
        # depth = F.interpolate(depth, size=[img_height, img_width])
        # depth = depth.squeeze(dim=1)  # [B, H, W]

        self.reconstr_loss = 0
        self.ssim_loss = 0
        # self.smooth_loss = 0

        warped_img_list = []
        mask_list = []
        reprojection_losses = []
        for view in range(1, num_views):
            view_img = imgs[view]
            view_cam = cams[view]
            # print('view_cam: {}'.format(view_cam.shape))
            # view_img = F.interpolate(view_img, scale_factor=0.25, mode='bilinear')
            if stage_idx == 0:
                view_img = F.interpolate(view_img, scale_factor=0.25,recompute_scale_factor=True)
            elif stage_idx == 1:
                view_img = F.interpolate(view_img, scale_factor=0.5,recompute_scale_factor=True)
            else:
                pass
            view_img = view_img.permute(0, 2, 3, 1)  # [B, C, H, W] --> [B, H, W, C]
            # warp view_img to the ref_img using the dmap of the ref_img
            warped_img, mask = inverse_warping(view_img, ref_cam, view_cam, depth)
            warped_img_list.append(warped_img)
            mask_list.append(mask)

            reconstr_loss = compute_reconstr_loss(warped_img, ref_img, mask, simple=False)
            valid_mask = 1 - mask  # replace all 0 values with INF
            reprojection_losses.append(reconstr_loss + 1e4 * valid_mask)

            # SSIM loss##
            if view < 3:
                self.ssim_loss += torch.mean(self.ssim(ref_img, warped_img, mask))

        ##smooth loss##
        # self.smooth_loss += depth_smoothness(depth.unsqueeze(dim=-1), ref_img, 1.0)

        # top-k operates along the last dimension, so swap the axes accordingly
        reprojection_volume = torch.stack(reprojection_losses).permute(1, 2, 3, 4, 0)
        # print('reprojection_volume: {}'.format(reprojection_volume.shape))
        # by default, it'll return top-k largest entries, hence sorted=False to get smallest entries
        # top_vals, top_inds = torch.topk(torch.neg(reprojection_volume), k=3, sorted=False)
        top_vals, top_inds = torch.topk(torch.neg(reprojection_volume), k=1, sorted=False)
        top_vals = torch.neg(top_vals)
        # top_mask = top_vals < (1e4 * torch.ones_like(top_vals, device=device))
        top_mask = top_vals < (1e4 * torch.ones_like(top_vals).cuda())
        top_mask = top_mask.float()
        top_vals = torch.mul(top_vals, top_mask)
        # print('top_vals: {}'.format(top_vals.shape))

        self.reconstr_loss = torch.mean(torch.sum(top_vals, dim=-1))
        self.unsup_loss = 12 * self.reconstr_loss + 6 * self.ssim_loss 
        # 按照un_mvsnet和M3VSNet的设置
        # self.unsup_loss = (0.8 * self.reconstr_loss + 0.2 * self.ssim_loss + 0.067 * self.smooth_loss) * 15
        return self.unsup_loss

class UnSupLoss_07(nn.Module):
    def __init__(self):
        super(UnSupLoss_07, self).__init__()
        self.ssim = SSIM()

    def forward(self, imgs, cams, depth, stage_idx):
        # print('imgs: {}'.format(imgs.shape))
        # print('cams: {}'.format(cams.shape))
        # print('depth: {}'.format(depth.shape))

        imgs = torch.unbind(imgs, 1)
        cams = torch.unbind(cams, 1)
        assert len(imgs) == len(cams), "Different number of images and projection matrices"
        img_height, img_width = imgs[0].shape[2], imgs[0].shape[3]
        num_views = len(imgs)

        ref_img = imgs[0]
        # ref_img = F.interpolate(ref_img, scale_factor=0.25, mode='bilinear')
        # 按照stage进行resize，匹配到每个阶段的分辨率
        # 这里尽量不要使用bilinear，这个会平滑图像的边缘，可能会对自监督损失有影响
        if stage_idx == 0:
            ref_img = F.interpolate(ref_img, scale_factor=0.25,recompute_scale_factor=True)
        elif stage_idx == 1:
            ref_img = F.interpolate(ref_img, scale_factor=0.5,recompute_scale_factor=True)
        else:
            pass
        ref_img = ref_img.permute(0, 2, 3, 1)  # [B, C, H, W] --> [B, H, W, C]
        ref_cam = cams[0]
        # print('ref_cam: {}'.format(ref_cam.shape))

        # depth reshape
        # depth = depth.unsqueeze(dim=1)  # [B, 1, H, W]
        # depth = F.interpolate(depth, size=[img_height, img_width])
        # depth = depth.squeeze(dim=1)  # [B, H, W]

        self.reconstr_loss = 0
        self.ssim_loss = 0
        self.smooth_loss = 0

        warped_img_list = []
        mask_list = []
        reprojection_losses = []
        for view in range(1, num_views):
            view_img = imgs[view]
            view_cam = cams[view]
            # print('view_cam: {}'.format(view_cam.shape))
            # view_img = F.interpolate(view_img, scale_factor=0.25, mode='bilinear')
            if stage_idx == 0:
                view_img = F.interpolate(view_img, scale_factor=0.25,recompute_scale_factor=True)
            elif stage_idx == 1:
                view_img = F.interpolate(view_img, scale_factor=0.5,recompute_scale_factor=True)
            else:
                pass
            view_img = view_img.permute(0, 2, 3, 1)  # [B, C, H, W] --> [B, H, W, C]
            # warp view_img to the ref_img using the dmap of the ref_img
            warped_img, mask = inverse_warping(view_img, ref_cam, view_cam, depth)
            warped_img_list.append(warped_img)
            mask_list.append(mask)

            reconstr_loss = compute_reconstr_loss(warped_img, ref_img, mask, simple=False)
            valid_mask = 1 - mask  # replace all 0 values with INF
            reprojection_losses.append(reconstr_loss + 1e4 * valid_mask)

            # SSIM loss##
            if view < 3:
                self.ssim_loss += torch.mean(self.ssim(ref_img, warped_img, mask))

        ##smooth loss##
        self.smooth_loss += depth_smoothness(depth.unsqueeze(dim=-1), ref_img, 1.0)

        # top-k operates along the last dimension, so swap the axes accordingly
        reprojection_volume = torch.stack(reprojection_losses).permute(1, 2, 3, 4, 0)
        # print('reprojection_volume: {}'.format(reprojection_volume.shape))
        # by default, it'll return top-k largest entries, hence sorted=False to get smallest entries
        # top_vals, top_inds = torch.topk(torch.neg(reprojection_volume), k=3, sorted=False)
        top_vals, top_inds = torch.topk(torch.neg(reprojection_volume), k=1, sorted=False)
        top_vals = torch.neg(top_vals)
        # top_mask = top_vals < (1e4 * torch.ones_like(top_vals, device=device))
        top_mask = top_vals < (1e4 * torch.ones_like(top_vals).cuda())
        top_mask = top_mask.float()
        top_vals = torch.mul(top_vals, top_mask)
        # print('top_vals: {}'.format(top_vals.shape))

        self.reconstr_loss = torch.mean(torch.sum(top_vals, dim=-1))
        self.unsup_loss = 12 * self.reconstr_loss + 6 * self.ssim_loss + 0.19 * self.smooth_loss
        # 按照un_mvsnet和M3VSNet的设置
        # self.unsup_loss = (0.8 * self.reconstr_loss + 0.2 * self.ssim_loss + 0.067 * self.smooth_loss) * 15
        return self.unsup_loss

class UnSupLoss_06(nn.Module):
    def __init__(self):
        super(UnSupLoss_06, self).__init__()
        self.ssim = SSIM()

    def forward(self, imgs, cams, depth, stage_idx):
        # print('imgs: {}'.format(imgs.shape))
        # print('cams: {}'.format(cams.shape))
        # print('depth: {}'.format(depth.shape))

        imgs = torch.unbind(imgs, 1)
        cams = torch.unbind(cams, 1)
        assert len(imgs) == len(cams), "Different number of images and projection matrices"
        img_height, img_width = imgs[0].shape[2], imgs[0].shape[3]
        num_views = len(imgs)

        ref_img = imgs[0]
        # ref_img = F.interpolate(ref_img, scale_factor=0.25, mode='bilinear')
        # 按照stage进行resize，匹配到每个阶段的分辨率
        # 这里尽量不要使用bilinear，这个会平滑图像的边缘，可能会对自监督损失有影响
        if stage_idx == 0:
            ref_img = F.interpolate(ref_img, scale_factor=0.25,recompute_scale_factor=True)
        elif stage_idx == 1:
            ref_img = F.interpolate(ref_img, scale_factor=0.5,recompute_scale_factor=True)
        else:
            pass
        ref_img = ref_img.permute(0, 2, 3, 1)  # [B, C, H, W] --> [B, H, W, C]
        ref_cam = cams[0]
        # print('ref_cam: {}'.format(ref_cam.shape))

        # depth reshape
        # depth = depth.unsqueeze(dim=1)  # [B, 1, H, W]
        # depth = F.interpolate(depth, size=[img_height, img_width])
        # depth = depth.squeeze(dim=1)  # [B, H, W]

        self.reconstr_loss = 0
        self.ssim_loss = 0
        self.smooth_loss = 0

        warped_img_list = []
        mask_list = []
        reprojection_losses = []
        for view in range(1, num_views):
            view_img = imgs[view]
            view_cam = cams[view]
            # print('view_cam: {}'.format(view_cam.shape))
            # view_img = F.interpolate(view_img, scale_factor=0.25, mode='bilinear')
            if stage_idx == 0:
                view_img = F.interpolate(view_img, scale_factor=0.25,recompute_scale_factor=True)
            elif stage_idx == 1:
                view_img = F.interpolate(view_img, scale_factor=0.5,recompute_scale_factor=True)
            else:
                pass
            view_img = view_img.permute(0, 2, 3, 1)  # [B, C, H, W] --> [B, H, W, C]
            # warp view_img to the ref_img using the dmap of the ref_img
            warped_img, mask = inverse_warping(view_img, ref_cam, view_cam, depth)
            warped_img_list.append(warped_img)
            mask_list.append(mask)

            reconstr_loss = compute_reconstr_loss(warped_img, ref_img, mask, simple=False)
            valid_mask = 1 - mask  # replace all 0 values with INF
            reprojection_losses.append(reconstr_loss + 1e4 * valid_mask)

            # SSIM loss##
            if view < 3:
                self.ssim_loss += torch.mean(self.ssim(ref_img, warped_img, mask))

        ##smooth loss##
        self.smooth_loss += depth_smoothness(depth.unsqueeze(dim=-1), ref_img, 1.0)

        # top-k operates along the last dimension, so swap the axes accordingly
        reprojection_volume = torch.stack(reprojection_losses).permute(1, 2, 3, 4, 0)
        # print('reprojection_volume: {}'.format(reprojection_volume.shape))
        # by default, it'll return top-k largest entries, hence sorted=False to get smallest entries
        # top_vals, top_inds = torch.topk(torch.neg(reprojection_volume), k=3, sorted=False)
        top_vals, top_inds = torch.topk(torch.neg(reprojection_volume), k=1, sorted=False)
        top_vals = torch.neg(top_vals)
        # top_mask = top_vals < (1e4 * torch.ones_like(top_vals, device=device))
        top_mask = top_vals < (1e4 * torch.ones_like(top_vals).cuda())
        top_mask = top_mask.float()
        top_vals = torch.mul(top_vals, top_mask)
        # print('top_vals: {}'.format(top_vals.shape))

        self.reconstr_loss = torch.mean(torch.sum(top_vals, dim=-1))
        self.unsup_loss = 12 * self.reconstr_loss + 6 * self.ssim_loss + 0.16 * self.smooth_loss
        # 按照un_mvsnet和M3VSNet的设置
        # self.unsup_loss = (0.8 * self.reconstr_loss + 0.2 * self.ssim_loss + 0.067 * self.smooth_loss) * 15
        return self.unsup_loss

class UnsupLossMultiStage_06(nn.Module):
    def __init__(self):
        super(UnsupLossMultiStage_06, self).__init__()
        self.unsup_loss = UnSupLoss_06()

    def forward(self, inputs, imgs, cams, **kwargs):
        depth_loss_weights = kwargs.get("dlossw", None)

        total_loss = torch.tensor(0.0, dtype=torch.float32, device=imgs.device, requires_grad=False)

        scalar_outputs = {}
        for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
            stage_idx = int(stage_key.replace("stage", "")) - 1

            depth_est = stage_inputs["depth"]
            depth_loss = self.unsup_loss(imgs, cams[stage_key], depth_est, stage_idx)


            if depth_loss_weights is not None:
                total_loss += depth_loss_weights[stage_idx] * depth_loss
            else:
                total_loss += 1.0 * depth_loss

            scalar_outputs["depth_loss_stage{}".format(stage_idx + 1)] = depth_loss
            scalar_outputs["reconstr_loss_stage{}".format(stage_idx + 1)] = self.unsup_loss.reconstr_loss
            scalar_outputs["ssim_loss_stage{}".format(stage_idx + 1)] = self.unsup_loss.ssim_loss
            scalar_outputs["smooth_loss_stage{}".format(stage_idx + 1)] = self.unsup_loss.smooth_loss

        return total_loss, scalar_outputs

class UnsupLossMultiStage_07(nn.Module):
    def __init__(self):
        super(UnsupLossMultiStage_07, self).__init__()
        self.unsup_loss = UnSupLoss_07()

    def forward(self, inputs, imgs, cams, **kwargs):
        depth_loss_weights = kwargs.get("dlossw", None)

        total_loss = torch.tensor(0.0, dtype=torch.float32, device=imgs.device, requires_grad=False)

        scalar_outputs = {}
        for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
            stage_idx = int(stage_key.replace("stage", "")) - 1

            depth_est = stage_inputs["depth"]
            depth_loss = self.unsup_loss(imgs, cams[stage_key], depth_est, stage_idx)


            if depth_loss_weights is not None:
                total_loss += depth_loss_weights[stage_idx] * depth_loss
            else:
                total_loss += 1.0 * depth_loss

            scalar_outputs["depth_loss_stage{}".format(stage_idx + 1)] = depth_loss
            scalar_outputs["reconstr_loss_stage{}".format(stage_idx + 1)] = self.unsup_loss.reconstr_loss
            scalar_outputs["ssim_loss_stage{}".format(stage_idx + 1)] = self.unsup_loss.ssim_loss
            scalar_outputs["smooth_loss_stage{}".format(stage_idx + 1)] = self.unsup_loss.smooth_loss

        return total_loss, scalar_outputs

class UnsupLossMultiStage(nn.Module):
    def __init__(self):
        super(UnsupLossMultiStage, self).__init__()
        self.unsup_loss = UnSupLoss()

    def forward(self, inputs, imgs, cams, **kwargs):
        depth_loss_weights = kwargs.get("dlossw", None)

        total_loss = torch.tensor(0.0, dtype=torch.float32, device=imgs.device, requires_grad=False)

        scalar_outputs = {}
        for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
            stage_idx = int(stage_key.replace("stage", "")) - 1

            depth_est = stage_inputs["depth"]
            depth_loss = self.unsup_loss(imgs, cams[stage_key], depth_est, stage_idx)


            if depth_loss_weights is not None:
                total_loss += depth_loss_weights[stage_idx] * depth_loss
            else:
                total_loss += 1.0 * depth_loss

            scalar_outputs["depth_loss_stage{}".format(stage_idx + 1)] = depth_loss
            scalar_outputs["reconstr_loss_stage{}".format(stage_idx + 1)] = self.unsup_loss.reconstr_loss
            scalar_outputs["ssim_loss_stage{}".format(stage_idx + 1)] = self.unsup_loss.ssim_loss
            scalar_outputs["smooth_loss_stage{}".format(stage_idx + 1)] = self.unsup_loss.smooth_loss

        return total_loss, scalar_outputs

class UnsupLossMultiStage_no_smooth(nn.Module):
    def __init__(self):
        super(UnsupLossMultiStage_no_smooth, self).__init__()
        self.unsup_loss = UnSupLoss_no_smooth()

    def forward(self, inputs, imgs, cams, **kwargs):
        depth_loss_weights = kwargs.get("dlossw", None)

        total_loss = torch.tensor(0.0, dtype=torch.float32, device=imgs.device, requires_grad=False)

        scalar_outputs = {}
        for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
            stage_idx = int(stage_key.replace("stage", "")) - 1

            depth_est = stage_inputs["depth"]
            depth_loss = self.unsup_loss(imgs, cams[stage_key], depth_est, stage_idx)


            if depth_loss_weights is not None:
                total_loss += depth_loss_weights[stage_idx] * depth_loss
            else:
                total_loss += 1.0 * depth_loss

            scalar_outputs["depth_loss_stage{}".format(stage_idx + 1)] = depth_loss
            scalar_outputs["reconstr_loss_stage{}".format(stage_idx + 1)] = self.unsup_loss.reconstr_loss
            scalar_outputs["ssim_loss_stage{}".format(stage_idx + 1)] = self.unsup_loss.ssim_loss
            # scalar_outputs["smooth_loss_stage{}".format(stage_idx + 1)] = self.unsup_loss.smooth_loss

        return total_loss, scalar_outputs
