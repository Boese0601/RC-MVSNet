from torch.utils.data import Dataset
import numpy as np
import os, cv2, time, math
from PIL import Image
from torchvision import transforms

from datasets.data_io import *
from datasets.utils import *


# the DTU dataset preprocessed by Yao Yao (only for training)
class MVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, ndepths=192, interval_scale=1.06,random_view=False,**kwargs):
        super(MVSDataset, self).__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.ndepths = ndepths
        self.interval_scale = interval_scale
        self.kwargs = kwargs
        self.random_view = random_view
        # self.scale_factor = 1
        print("mvsdataset kwargs", self.kwargs)

        assert self.mode in ["train", "val", "test"]
        self.metas = self.build_list()
        self.define_transforms()
        self.build_proj_mats()
    
    def build_proj_mats(self):
        proj_mats, intrinsics_nerf, world2cams, cam2worlds = [], [], [], []
        for vid in self.id_list:
            proj_mat_filename = os.path.join(self.datapath,
                                             f'Cameras/train/{vid:08d}_cam.txt')
            intrinsic_nerf, extrinsic,_, _, near_far = self.read_cam_file(proj_mat_filename)
            intrinsic_nerf[:2] *= 4
            extrinsic[:3, 3] # *= self.scale_factor

            # intrinsic[:2] = intrinsic[:2] * self.downSample
            intrinsics_nerf += [intrinsic_nerf.copy()]

            # multiply intrinsics and extrinsics to get projection matrix
            proj_mat_l = np.eye(4)
            intrinsic_nerf[:2] = intrinsic_nerf[:2] / 4
            proj_mat_l[:3, :4] = intrinsic_nerf @ extrinsic[:3, :4]

            proj_mats += [(proj_mat_l, near_far)]
            world2cams += [extrinsic]
            cam2worlds += [np.linalg.inv(extrinsic)]

        self.proj_mats, self.intrinsics_nerf = np.stack(proj_mats), np.stack(intrinsics_nerf)
        self.world2cams, self.cam2worlds = np.stack(world2cams), np.stack(cam2worlds)
    
    def build_list(self):
        metas = []
        with open(self.listfile) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

        self.id_list = []
        # scans
        for scan in scans:
            pair_file = "Cameras/pair.txt"
            # read the pair file
            with open(os.path.join(self.datapath, pair_file)) as f:
                num_viewpoint = int(f.readline())
                # viewpoints (49)
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    # light conditions 0-6
                    # light conditions 0-6 for training
                    # light condition 3 for testing (the brightest?)
                    for light_idx in range(7):
                        metas.append((scan, light_idx, ref_view, src_views))
                        self.id_list.append([ref_view] + src_views)
        self.id_list = np.unique(self.id_list)
        self.build_remap()
        print("dataset", self.mode, "metas:", len(metas))
        return metas

    def build_remap(self):
        self.remap = np.zeros(np.max(self.id_list) + 1).astype('int')
        for i, item in enumerate(self.id_list):
            self.remap[item] = i


    def define_transforms(self):
        # self.transform_aug = transforms.Compose([
        #     transforms.ColorJitter(brightness=1, contrast=1, saturation=0.5, hue=0.5),
        #     transforms.ToTensor(),
        #     RandomGamma(min_gamma=0.5, max_gamma=2.0, clip_image=True),
        #     # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # ])
        # self.transform_seg = transforms.Compose([
        #     transforms.ToTensor(),
        #     # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # ])
        self.transform_aug = transforms.Compose([
            transforms.ColorJitter(brightness=1, contrast=1, saturation=0.5, hue=0.5),
            transforms.ToTensor(),
            RandomGamma(min_gamma=0.5, max_gamma=2.0, clip_image=True),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.transform_seg = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


    def __len__(self):
        return len(self.metas)


    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0]) 
        depth_interval = float(lines[11].split()[1]) * self.interval_scale 
        depth_max = depth_min + depth_interval * self.ndepths
        return intrinsics, extrinsics, depth_min, depth_interval, [depth_min, depth_max]


    def read_img(self, filename):
        img = Image.open(filename)
        # scale 0~255 to 0~1
        np_img = np.array(img, dtype=np.float32) / 255.
        return np_img


    # def read_img_aug(self, filename):
    #     img = Image.open(filename)
    #     # scale 0~255 to 0~1
    #     np_img = np.array(img, dtype=np.float32) / 255.
    #     tr_img = self.transform_aug(np_img)
    #     return tr_img

    def read_img_seg(self, filename):
        img = Image.open(filename)
        return self.transform_seg(img)


    def read_img_aug(self, filename):
        img = Image.open(filename)
        img = self.transform_aug(img)
        return img


    def center_image(self, img):
        """ normalize image input """
        img = img.astype(np.float32)
        var = np.var(img, axis=(0, 1), keepdims=True)
        mean = np.mean(img, axis=(0, 1), keepdims=True)
        return (img - mean) / (np.sqrt(var) + 0.00000001)


    def prepare_img(self, hr_img):
        # w1600-h1200-> 800-600 ; crop -> 640, 512; downsample 1/4 -> 160, 128

        # downsample
        h, w = hr_img.shape
        hr_img_ds = cv2.resize(hr_img, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)
        # crop
        h, w = hr_img_ds.shape
        target_h, target_w = 512, 640
        start_h, start_w = (h - target_h) // 2, (w - target_w) // 2
        hr_img_crop = hr_img_ds[start_h: start_h + target_h, start_w: start_w + target_w]

        # #downsample
        # lr_img = cv2.resize(hr_img_crop, (target_w//4, target_h//4), interpolation=cv2.INTER_NEAREST)

        return hr_img_crop


    def read_mask_hr(self, filename):
        img = Image.open(filename)
        np_img = np.array(img, dtype=np.float32)
        np_img = (np_img > 10).astype(np.float32)
        np_img = self.prepare_img(np_img)

        h, w = np_img.shape
        np_img_ms = {
            "stage1": cv2.resize(np_img, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(np_img, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST),
            "stage3": np_img,
        }
        return np_img_ms

    def read_depth_all(self, filename):
        depth_h = np.array(read_pfm(filename)[0], dtype=np.float32)  # (800, 800)
        depth_h = cv2.resize(depth_h, None, fx=0.5, fy=0.5,
                             interpolation=cv2.INTER_NEAREST)  # (600, 800)
        depth_h = depth_h[44:556, 80:720]  # (512, 640)
        depth_h = cv2.resize(depth_h, None, fx=1.0, fy=1.0,
                             interpolation=cv2.INTER_NEAREST)  # !!!!!!!!!!!!!!!!!!!!!!!!!
        # depth = cv2.resize(depth_h, None, fx=1.0 / 4, fy=1.0 / 4,
        #                    interpolation=cv2.INTER_NEAREST)  # !!!!!!!!!!!!!!!!!!!!!!!!!

        return depth_h

    def read_depth(self, filename):
        # read pfm depth file
        return np.array(read_pfm(filename)[0], dtype=np.float32)


    def read_depth_hr(self, filename):
        # read pfm depth file
        # w1600-h1200-> 800-600 ; crop -> 640, 512; downsample 1/4 -> 160, 128
        depth_hr = np.array(read_pfm(filename)[0], dtype=np.float32)
        depth_lr = self.prepare_img(depth_hr)

        h, w = depth_lr.shape
        depth_lr_ms = {
            "stage1": cv2.resize(depth_lr, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(depth_lr, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST),
            "stage3": depth_lr,
        }
        return depth_lr_ms

    def __getitem__(self, idx):
        sample = {}
        meta = self.metas[idx]
        scan, light_idx, ref_view, src_views = meta
        # use only the reference view and first nviews-1 source views
        if not self.random_view:
            view_ids = [ref_view] + src_views[:self.nviews - 1]
        else:
            num_src_views = len(src_views)
            rand_ids = torch.randperm(num_src_views)[:self.nviews - 1]
            src_views_t = torch.tensor(src_views)
            view_ids = [ref_view] + list(src_views_t[rand_ids].numpy())    
        imgs = []
        imgs_aug = []
        center_imgs = []
        mask = None
        depth_values = None
        proj_matrices = []

        affine_mat, affine_mat_inv = [], []
        depths_h = []
        proj_mats, intrinsics_all, w2cs, c2ws, near_fars = [], [], [], [], [] 
        for i, vid in enumerate(view_ids):
            # NOTE that the id in image file names is from 1 to 49 (not 0~48)
            img_filename = os.path.join(self.datapath,
                                        'Rectified/{}_train/rect_{:0>3}_{}_r5000.png'.format(scan, vid + 1, light_idx))

            mask_filename_hr = os.path.join(self.datapath, 'Depths_raw/{}/depth_visual_{:0>4}.png'.format(scan, vid))
            depth_filename_hr = os.path.join(self.datapath, 'Depths_raw/{}/depth_map_{:0>4}.pfm'.format(scan, vid))

            proj_mat_filename = os.path.join(self.datapath, 'Cameras/train/{:0>8}_cam.txt').format(vid)

            # img = self.read_img(img_filename)
            image_aug = self.read_img_aug(img_filename)
            image_seg = self.read_img_seg(img_filename)
            center_img = self.center_image(cv2.cvtColor(cv2.imread(img_filename), cv2.COLOR_BGR2RGB))


            # Nerf_data
            index_mat = self.remap[vid]
            proj_mat_ls, near_far = self.proj_mats[index_mat]
            intrinsics_all.append(self.intrinsics_nerf[index_mat])
            w2cs.append(self.world2cams[index_mat])
            c2ws.append(self.cam2worlds[index_mat])

            affine_mat.append(proj_mat_ls)
            affine_mat_inv.append(np.linalg.inv(proj_mat_ls))
            
            if i == 0:  # reference view
                ref_proj_inv = np.linalg.inv(proj_mat_ls)
                proj_mats += [np.eye(4)]
            else:
                proj_mats += [proj_mat_ls @ ref_proj_inv]

            if os.path.exists(depth_filename_hr):
                depth_h = self.read_depth_all(depth_filename_hr)
                # depth_h *= self.scale_factor
                depths_h.append(depth_h)
            else:
                depths_h.append(np.zeros((1, 1)))

            near_fars.append(near_far)

            ##################################################################################################

            intrinsics, extrinsics, depth_min, depth_interval, _ = self.read_cam_file(proj_mat_filename)

            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics

            proj_matrices.append(proj_mat)

            if i == 0:  # reference view
                mask_read_ms = self.read_mask_hr(mask_filename_hr)
                depth_ms = self.read_depth_hr(depth_filename_hr)

                # get depth values
                depth_max = depth_interval * self.ndepths + depth_min
                depth_values = np.arange(depth_min, depth_max, depth_interval, dtype=np.float32)

                mask = mask_read_ms

            # imgs.append(img)
            imgs.append(image_seg)
            imgs_aug.append(image_aug)
            center_imgs.append(center_img)

        # all
        # imgs = np.stack(imgs).transpose([0, 3, 1, 2])
        imgs = np.stack(imgs)
        center_imgs = np.stack(center_imgs).transpose([0, 3, 1, 2])
        imgs_aug = torch.stack(imgs_aug)


        # Nerf Data 
        depths_h = np.stack(depths_h)
        proj_mats = np.stack(proj_mats)[:, :3]
        affine_mat, affine_mat_inv = np.stack(affine_mat), np.stack(affine_mat_inv)
        intrinsics_all, w2cs, c2ws, near_fars = np.stack(intrinsics_all), np.stack(w2cs), np.stack(c2ws), np.stack(near_fars)
        view_ids_all = [ref_view] + list(src_views) if type(src_views[0]) is not list else [j for sub in src_views for j in sub]
        c2ws_all = self.cam2worlds[self.remap[view_ids_all]]
        ###########################################################

        # ms proj_mats
        proj_matrices = np.stack(proj_matrices)
        stage2_pjmats = proj_matrices.copy()
        stage2_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 2
        stage3_pjmats = proj_matrices.copy()
        stage3_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 4

        proj_matrices_ms = {
            "stage1": proj_matrices,
            "stage2": stage2_pjmats,
            "stage3": stage3_pjmats
        }

        sample["imgs"] = imgs
        sample["imgs_aug"] = imgs_aug
        sample["proj_matrices"] = proj_matrices_ms
        sample["depth"] = depth_ms
        sample["depth_values"] = depth_values
        sample["mask"] = mask
        sample["center_imgs"] = center_imgs

        # nerf_render_data
        sample['depths_h'] = depths_h.astype(np.float32)  # (V, H, W)
        sample['w2cs'] = w2cs.astype(np.float32)  # (V, 4, 4)
        sample['c2ws'] = c2ws.astype(np.float32)  # (V, 4, 4)
        sample['near_fars'] = near_fars.astype(np.float32)
        sample['proj_mats'] = proj_mats.astype(np.float32)
        sample['intrinsics'] = intrinsics_all.astype(np.float32)  # (V, 3, 3)
        sample['view_ids'] = np.array(view_ids)
        sample['light_id'] = np.array(light_idx)
        sample['affine_mat'] = affine_mat
        sample['affine_mat_inv'] = affine_mat_inv
        sample['scan'] = scan
        sample['c2ws_all'] = c2ws_all.astype(np.float32)
        return sample