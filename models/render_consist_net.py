from losses.sl1loss import SL1Loss
from .render_models import *
from .renderer import *
from .render_utils import *
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Rendering_Consistency_Net(nn.Module):
    def __init__(self, args):
        super(Rendering_Consistency_Net, self).__init__()
        self.args = args
        self.args.feat_dim = 8+3*4
        self.idx = 0

        # Create nerf model
        self.render_kwargs_train, self.render_kwargs_test, start, self.grad_vars = create_nerf_mvs(args, use_mvs=True, dir_embedder=False, pts_embedder=True,share_warp=True)
        filter_keys(self.render_kwargs_train)

        # Create mvs model
        self.MVSNet = self.render_kwargs_train['network_mvs']
        self.network_fn = self.render_kwargs_train['network_fn']
        self.network_query_fn = self.render_kwargs_train['network_query_fn']
        self.white_bkgd = self.render_kwargs_train['white_bkgd']
        self.render_kwargs_train.pop('network_fn')
        self.render_kwargs_train.pop('network_query_fn')
        self.render_kwargs_train.pop('white_bkgd')
        self.render_kwargs_train.pop('network_mvs')
        self.render_kwargs_train['NDC_local'] = False




    def decode_batch(self, batch, idx=list(torch.arange(4))):

        data_mvs = sub_selete_data(batch, device, idx, filtKey=[])
        pose_ref = {'w2cs': data_mvs['w2cs'].squeeze(), 'intrinsics': data_mvs['intrinsics'].squeeze(),
                    'c2ws': data_mvs['c2ws'].squeeze(),'near_fars':data_mvs['near_fars'].squeeze()}

        return data_mvs, pose_ref

    def unpreprocess(self, data, shape=(1,1,3,1,1)):
        # to unnormalize image for visualization
        # data N V C H W
        device = data.device
        mean = torch.tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]).view(*shape).to(device)
        std = torch.tensor([1 / 0.229, 1 / 0.224, 1 / 0.225]).view(*shape).to(device)

        return (data - mean) / std


    def forward(self, volume_feature_warp,pseudo_depth, batch):
        if 'scan' in batch.keys():
            batch.pop('scan')
        data_mvs, pose_ref = self.decode_batch(batch)
        imgs, proj_mats = data_mvs['imgs'], data_mvs['proj_mats']
        near_fars, depths_h = data_mvs['near_fars'], data_mvs['depths_h']
        _,V,H,W = depths_h.shape

        pseudo_depth_h = pseudo_depth.expand(V,H,W).unsqueeze(0)

        volume_feature = self.MVSNet(volume_feature_warp,pad=self.args.pad)
        imgs = self.unpreprocess(imgs)


        N_rays, N_samples = 1024, self.args.N_samples   
        c2ws, w2cs, intrinsics = pose_ref['c2ws'], pose_ref['w2cs'], pose_ref['intrinsics']

        rays_pts, rays_dir, target_s, rays_NDC, depth_candidates, rays_o, rays_depth, ndc_parameters = \
                build_rays_norm(imgs, pseudo_depth_h, pose_ref, w2cs, c2ws, intrinsics, near_fars, N_rays, N_samples, pad=self.args.pad)
        rgb, disp, acc, depth_pred, alpha, ret = rendering(self.args, pose_ref, rays_pts, rays_NDC, depth_candidates, rays_o, rays_dir,
                                                       volume_feature, imgs[:, -3:], img_feat=None,network_fn=self.network_fn,network_query_fn=self.network_query_fn,white_bkgd=self.white_bkgd)

        return rgb, disp, acc, depth_pred, alpha, ret, rays_depth, target_s
