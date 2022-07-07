import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import *
from typing import List, Tuple, Dict, Callable, Any, Union
from collections import OrderedDict

Align_Corners_Range = False

class ListModule(nn.Module):
    def __init__(self, modules: Union[List, OrderedDict]):
        super(ListModule, self).__init__()
        if isinstance(modules, OrderedDict):
            iterable = modules.items()
        elif isinstance(modules, list):
            iterable = enumerate(modules)
        else:
            raise TypeError('modules should be OrderedDict or List.')
        for name, module in iterable:
            if not isinstance(module, nn.Module):
                module = ListModule(module)
            if not isinstance(name, str):
                name = str(name)
            self.add_module(name, module)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)


def entropy(volume, dim, keepdim=False):
    return torch.sum(-volume * volume.clamp(1e-9, 1.).log(), dim=dim, keepdim=keepdim)

class DepthNet(nn.Module):
    def __init__(self):
        super(DepthNet, self).__init__()

    def forward(self, features, proj_matrices, depth_values, num_depth, cost_regularization,imgs,pad=0, prob_volume_init=None):
        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(features) == len(proj_matrices), "Different number of images and projection matrices"
        assert depth_values.shape[1] == num_depth, "depth_values.shape[1]:{}  num_depth:{}".format(
            depth_values.shapep[1], num_depth)
        num_views = len(features)
        B = imgs.shape[0]
        _,C,H,W = features[0].shape
        ref_feature, src_features = features[0], features[1:]
        img_feat = torch.empty((B, 3*num_views + C, num_depth, *ref_feature.shape[-2:]), device=imgs.device, dtype=torch.float)
        img_feat_no_ref = torch.empty((B, 3*(num_views-1) + C, num_depth, *ref_feature.shape[-2:]), device=imgs.device, dtype=torch.float)
        imgs = F.interpolate(imgs.view(B * num_views, *imgs.shape[2:]), (H, W), mode='bilinear', align_corners=False).view(B, num_views,-1,H,W).permute(1, 0, 2, 3, 4)
        img_feat[:, :3, :, pad:H + pad, pad:W + pad] = imgs[0].unsqueeze(2).expand(-1, -1, num_depth, -1, -1)
        # step 1. feature extraction
        # in: images; out: 32-channel feature maps
        
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

        # step 2. differentiable homograph, build cost volume
        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
        volume_sum = ref_volume
        volume_sq_sum = ref_volume ** 2
        volume_sum_no_ref = 0
        volume_sq_sum_no_ref = 0


        del ref_volume
        for i,(src_img, src_fea, src_proj) in enumerate(zip(imgs[1:],src_features, src_projs)):
            # warpped features
            src_proj_new = src_proj[:, 0].clone()
            src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
            ref_proj_new = ref_proj[:, 0].clone()
            ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
            img_feat[:, (i + 1) * 3:(i + 2) * 3] = homo_warping(src_img, src_proj_new, ref_proj_new, depth_values)
            img_feat_no_ref[:, i * 3:(i + 1) * 3] = homo_warping(src_img, src_proj_new, ref_proj_new, depth_values)
            warped_volume = homo_warping(src_fea, src_proj_new, ref_proj_new, depth_values)

            if self.training:
                volume_sum = volume_sum + warped_volume
                volume_sq_sum = volume_sq_sum + warped_volume ** 2
                volume_sum_no_ref = volume_sum_no_ref + warped_volume
                volume_sq_sum_no_ref = volume_sq_sum_no_ref + warped_volume ** 2
            else:
                # TODO: this is only a temporal solution to save memory, better way?
                volume_sum += warped_volume
                volume_sq_sum += warped_volume.pow_(2)  # the memory of warped_volume has been modified
                volume_sum_no_ref = volume_sum_no_ref + warped_volume
                volume_sq_sum_no_ref = volume_sq_sum_no_ref + warped_volume ** 2
            del warped_volume
        # aggregate multiple feature volumes by variance
        volume_variance = volume_sq_sum.div_(num_views).sub_(volume_sum.div_(num_views).pow_(2))
        volume_variance_no_ref = volume_sq_sum_no_ref.div_(num_views).sub_(volume_sum_no_ref.div_(num_views).pow_(2))
        img_feat[:, -C:] = volume_variance
        img_feat_no_ref[:, -C:] = volume_variance_no_ref
        # step 3. cost volume regularization
        cost_reg = cost_regularization(volume_variance)
        # cost_reg = F.upsample(cost_reg, [num_depth * 4, img_height, img_width], mode='trilinear')
        prob_volume_pre = cost_reg.squeeze(1)

        if prob_volume_init is not None:
            prob_volume_pre += prob_volume_init

        prob_volume = F.softmax(prob_volume_pre, dim=1)
        depth = depth_regression(prob_volume, depth_values=depth_values)
        # entropy_ = entropy(prob_volume, dim=1, keepdim=True)
        # heads = self.uncert_net(entropy_)
        with torch.no_grad():
            # photometric confidence
            prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1),
                                                stride=1, padding=0).squeeze(1)
            depth_index = depth_regression(prob_volume, depth_values=torch.arange(num_depth, device=prob_volume.device,
                                                                                  dtype=torch.float)).long()
            depth_index = depth_index.clamp(min=0, max=num_depth - 1)
            photometric_confidence = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)

        return {"depth": depth, "photometric_confidence": photometric_confidence,"volume_feature_no_ref":img_feat_no_ref}

class CascadeMVSNet(nn.Module):
    def __init__(self, refine=False, ndepths=[48, 32, 8], depth_interals_ratio=[4, 2, 1], share_cr=False,
                 grad_method="detach", arch_mode="fpn", cr_base_chs=[8, 8, 8]):
        super(CascadeMVSNet, self).__init__()
        self.refine = refine
        self.share_cr = share_cr
        self.ndepths = ndepths
        self.depth_interals_ratio = depth_interals_ratio
        self.grad_method = grad_method
        self.arch_mode = arch_mode
        self.cr_base_chs = cr_base_chs
        self.num_stage = len(ndepths)
        print("**********netphs:{}, depth_intervals_ratio:{},  grad:{}, chs:{}************".format(ndepths,
                                                                                                   depth_interals_ratio,
                                                                                                   self.grad_method,
                                                                                                   self.cr_base_chs))

        assert len(ndepths) == len(depth_interals_ratio)

        self.stage_infos = {
            "stage1": {
                "scale": 4.0,
            },
            "stage2": {
                "scale": 2.0,
            },
            "stage3": {
                "scale": 1.0,
            }
        }
        self.feature = FeatureNet(base_channels=8, stride=4, num_stage=self.num_stage, arch_mode=self.arch_mode)


            
        if self.share_cr:
            self.cost_regularization = CostRegNet(in_channels=self.feature.out_channels, base_channels=8)
        else:
            self.cost_regularization = nn.ModuleList([CostRegNet(in_channels=self.feature.out_channels[i],
                                                                 base_channels=self.cr_base_chs[i])
                                                      for i in range(self.num_stage)])

        if self.refine:
            self.refine_network = RefineNet()
        self.DepthNet = DepthNet()

    def forward(self, imgs, proj_matrices, depth_values):
        depth_min = float(depth_values[0, 0].cpu().numpy())
        depth_max = float(depth_values[0, -1].cpu().numpy())
        depth_interval = (depth_max - depth_min) / depth_values.size(1)

        # step 1. feature extraction
        features = []
        for nview_idx in range(imgs.size(1)):  # imgs shape (B, N, C, H, W)
            img = imgs[:, nview_idx]
            features.append(self.feature(img))
        outputs = {}

        depth, cur_depth = None, None
        for stage_idx in range(self.num_stage):
            # print("*********************stage{}*********************".format(stage_idx + 1))
            # stage feature, proj_mats, scales
            features_stage = [feat["stage{}".format(stage_idx + 1)] for feat in features]
            proj_matrices_stage = proj_matrices["stage{}".format(stage_idx + 1)]
            stage_scale = self.stage_infos["stage{}".format(stage_idx + 1)]["scale"]

            if depth is not None:
                if self.grad_method == "detach":
                    cur_depth = depth.detach()
                else:
                    cur_depth = depth
                cur_depth = F.interpolate(cur_depth.unsqueeze(1),
                                          [img.shape[2], img.shape[3]], mode='bilinear',
                                          align_corners=Align_Corners_Range).squeeze(1)
            else:
                cur_depth = depth_values
            depth_range_samples = get_depth_range_samples(cur_depth=cur_depth,
                                                          ndepth=self.ndepths[stage_idx],
                                                          depth_inteval_pixel=self.depth_interals_ratio[
                                                                                  stage_idx] * depth_interval,
                                                          dtype=img[0].dtype,
                                                          device=img[0].device,
                                                          shape=[img.shape[0], img.shape[2], img.shape[3]],
                                                          max_depth=depth_max,
                                                          min_depth=depth_min)

            outputs_stage = self.DepthNet(features_stage, proj_matrices_stage,
                                          depth_values=F.interpolate(depth_range_samples.unsqueeze(1),
                                                                     [self.ndepths[stage_idx],
                                                                      img.shape[2] // int(stage_scale),
                                                                      img.shape[3] // int(stage_scale)],
                                                                     mode='trilinear',
                                                                     align_corners=Align_Corners_Range).squeeze(1),
                                          num_depth=self.ndepths[stage_idx],
                                          cost_regularization=self.cost_regularization if self.share_cr else
                                          self.cost_regularization[stage_idx],imgs=imgs)

            depth = outputs_stage['depth']

            outputs["stage{}".format(stage_idx + 1)] = outputs_stage
            outputs.update(outputs_stage)
        # depth map refinement
        if self.refine:
            refined_depth = self.refine_network(torch.cat((imgs[:, 0], depth), 1))
            outputs["refined_depth"] = refined_depth

        return outputs,outputs["stage1"]["volume_feature_no_ref"]

# Model for testing
class DepthNet_eval(nn.Module):
    def __init__(self):
        super(DepthNet_eval, self).__init__()

    def forward(self, features, proj_matrices, depth_values, num_depth, cost_regularization,imgs,pad=0, prob_volume_init=None):
        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(features) == len(proj_matrices), "Different number of images and projection matrices"
        assert depth_values.shape[1] == num_depth, "depth_values.shape[1]:{}  num_depth:{}".format(
            depth_values.shapep[1], num_depth)
        num_views = len(features)
        B = imgs.shape[0]
        _,C,H,W = features[0].shape
        ref_feature, src_features = features[0], features[1:]
        img_feat = torch.empty((B, 3*num_views + C, num_depth, *ref_feature.shape[-2:]), device=imgs.device, dtype=torch.float)
        # img_feat_no_ref = torch.empty((B, 3*(num_views-1) + C, num_depth, *ref_feature.shape[-2:]), device=imgs.device, dtype=torch.float)
        imgs = F.interpolate(imgs.view(B * num_views, *imgs.shape[2:]), (H, W), mode='bilinear', align_corners=False).view(B, num_views,-1,H,W).permute(1, 0, 2, 3, 4)
        img_feat[:, :3, :, pad:H + pad, pad:W + pad] = imgs[0].unsqueeze(2).expand(-1, -1, num_depth, -1, -1)
        # step 1. feature extraction
        # in: images; out: 32-channel feature maps
        
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

        # step 2. differentiable homograph, build cost volume
        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
        volume_sum = ref_volume
        volume_sq_sum = ref_volume ** 2
        # volume_sum_no_ref = 0
        # volume_sq_sum_no_ref = 0


        del ref_volume
        for i,(src_img, src_fea, src_proj) in enumerate(zip(imgs[1:],src_features, src_projs)):
            # warpped features
            src_proj_new = src_proj[:, 0].clone()
            src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
            ref_proj_new = ref_proj[:, 0].clone()
            ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
            img_feat[:, (i + 1) * 3:(i + 2) * 3] = homo_warping(src_img, src_proj_new, ref_proj_new, depth_values)
            # img_feat_no_ref[:, i * 3:(i + 1) * 3] = homo_warping(src_img, src_proj_new, ref_proj_new, depth_values)
            warped_volume = homo_warping(src_fea, src_proj_new, ref_proj_new, depth_values)

            if self.training:
                volume_sum = volume_sum + warped_volume
                volume_sq_sum = volume_sq_sum + warped_volume ** 2
                # volume_sum_no_ref = volume_sum_no_ref + warped_volume
                # volume_sq_sum_no_ref = volume_sq_sum_no_ref + warped_volume ** 2
            else:
                # TODO: this is only a temporal solution to save memory, better way?
                volume_sum += warped_volume
                volume_sq_sum += warped_volume.pow_(2)  # the memory of warped_volume has been modified
                # volume_sum_no_ref = volume_sum_no_ref + warped_volume
                # volume_sq_sum_no_ref = volume_sq_sum_no_ref + warped_volume ** 2
            del warped_volume
        # aggregate multiple feature volumes by variance
        volume_variance = volume_sq_sum.div_(num_views).sub_(volume_sum.div_(num_views).pow_(2))
        # volume_variance_no_ref = volume_sq_sum_no_ref.div_(num_views).sub_(volume_sum_no_ref.div_(num_views).pow_(2))
        img_feat[:, -C:] = volume_variance
        # img_feat_no_ref[:, -C:] = volume_variance_no_ref
        # step 3. cost volume regularization
        cost_reg = cost_regularization(volume_variance)
        prob_volume_pre = cost_reg.squeeze(1)

        if prob_volume_init is not None:
            prob_volume_pre += prob_volume_init

        prob_volume = F.softmax(prob_volume_pre, dim=1)
        depth = depth_regression(prob_volume, depth_values=depth_values)

        with torch.no_grad():
            # photometric confidence
            prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1),
                                                stride=1, padding=0).squeeze(1)
            depth_index = depth_regression(prob_volume, depth_values=torch.arange(num_depth, device=prob_volume.device,
                                                                                  dtype=torch.float)).long()
            depth_index = depth_index.clamp(min=0, max=num_depth - 1)
            photometric_confidence = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)

        return {"depth": depth, "photometric_confidence": photometric_confidence}

class CascadeMVSNet_eval(nn.Module):
    def __init__(self, refine=False, ndepths=[48, 32, 8], depth_interals_ratio=[4, 2, 1], share_cr=False,
                 grad_method="detach", arch_mode="fpn", cr_base_chs=[8, 8, 8]):
        super(CascadeMVSNet_eval, self).__init__()
        self.refine = refine
        self.share_cr = share_cr
        self.ndepths = ndepths
        self.depth_interals_ratio = depth_interals_ratio
        self.grad_method = grad_method
        self.arch_mode = arch_mode
        self.cr_base_chs = cr_base_chs
        self.num_stage = len(ndepths)
        print("**********netphs:{}, depth_intervals_ratio:{},  grad:{}, chs:{}************".format(ndepths,
                                                                                                   depth_interals_ratio,
                                                                                                   self.grad_method,
                                                                                                   self.cr_base_chs))

        assert len(ndepths) == len(depth_interals_ratio)

        self.stage_infos = {
            "stage1": {
                "scale": 4.0,
            },
            "stage2": {
                "scale": 2.0,
            },
            "stage3": {
                "scale": 1.0,
            }
        }

        self.feature = FeatureNet(base_channels=8, stride=4, num_stage=self.num_stage, arch_mode=self.arch_mode)

        if self.share_cr:
            self.cost_regularization = CostRegNet(in_channels=self.feature.out_channels, base_channels=8)
        else:
            self.cost_regularization = nn.ModuleList([CostRegNet(in_channels=self.feature.out_channels[i],
                                                                 base_channels=self.cr_base_chs[i])
                                                      for i in range(self.num_stage)])
        if self.refine:
            self.refine_network = RefineNet()
        self.DepthNet = DepthNet_eval()

    def forward(self, imgs, proj_matrices, depth_values):
        depth_min = float(depth_values[0, 0].cpu().numpy())
        depth_max = float(depth_values[0, -1].cpu().numpy())
        depth_interval = (depth_max - depth_min) / depth_values.size(1)

        # step 1. feature extraction
        features = []
        # feat_ = []
        for nview_idx in range(imgs.size(1)):  # imgs shape (B, N, C, H, W)
            img = imgs[:, nview_idx]
            features.append(self.feature(img))

        outputs = {}

        depth, cur_depth = None, None
        for stage_idx in range(self.num_stage):
            # print("*********************stage{}*********************".format(stage_idx + 1))
            # stage feature, proj_mats, scales
            features_stage = [feat["stage{}".format(stage_idx + 1)] for feat in features]
            proj_matrices_stage = proj_matrices["stage{}".format(stage_idx + 1)]
            stage_scale = self.stage_infos["stage{}".format(stage_idx + 1)]["scale"]

            if depth is not None:
                if self.grad_method == "detach":
                    cur_depth = depth.detach()
                else:
                    cur_depth = depth
                cur_depth = F.interpolate(cur_depth.unsqueeze(1),
                                          [img.shape[2], img.shape[3]], mode='bilinear',
                                          align_corners=Align_Corners_Range).squeeze(1)
            else:
                cur_depth = depth_values
            depth_range_samples = get_depth_range_samples(cur_depth=cur_depth,
                                                          ndepth=self.ndepths[stage_idx],
                                                          depth_inteval_pixel=self.depth_interals_ratio[
                                                                                  stage_idx] * depth_interval,
                                                          dtype=img[0].dtype,
                                                          device=img[0].device,
                                                          shape=[img.shape[0], img.shape[2], img.shape[3]],
                                                          max_depth=depth_max,
                                                          min_depth=depth_min)

            outputs_stage = self.DepthNet(features_stage, proj_matrices_stage,
                                          depth_values=F.interpolate(depth_range_samples.unsqueeze(1),
                                                                     [self.ndepths[stage_idx],
                                                                      img.shape[2] // int(stage_scale),
                                                                      img.shape[3] // int(stage_scale)],
                                                                     mode='trilinear',
                                                                     align_corners=Align_Corners_Range).squeeze(1),
                                          num_depth=self.ndepths[stage_idx],
                                          cost_regularization=self.cost_regularization if self.share_cr else
                                          self.cost_regularization[stage_idx],imgs=imgs)

            depth = outputs_stage['depth']

            outputs["stage{}".format(stage_idx + 1)] = outputs_stage
            outputs.update(outputs_stage)
        # depth map refinement
        if self.refine:
            refined_depth = self.refine_network(torch.cat((imgs[:, 0], depth), 1))
            outputs["refined_depth"] = refined_depth

        return outputs