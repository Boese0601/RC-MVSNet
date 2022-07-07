from torch.utils.data import Dataset
import numpy as np
import os, cv2, time
from PIL import Image
from torchvision import transforms

from datasets.data_io import *


s_h, s_w = 0, 0
class MVSDataset(Dataset):
    def __init__(self, datapath, split='intermediate', nviews=3, img_wh=(1920, 1056), ndepths=192):
        super(MVSDataset, self).__init__()
        self.datapath = datapath
        self.img_wh = img_wh
        self.split = split
        self.nviews = nviews
        self.ndepths = ndepths
        self.build_metas()
        self.define_transforms()
        
    def build_metas(self):
        self.metas = []
        if self.split == 'intermediate':
            self.scans = ['Family', 'Francis', 'Horse', 'Lighthouse',
                          'M60', 'Panther', 'Playground', 'Train']
            
            
            self.image_sizes = {'Family': (1920, 1080),
                                'Francis': (1920, 1080),
                                'Horse': (1920, 1080),
                                'Lighthouse': (2048, 1080),
                                'M60': (2048, 1080),
                                'Panther': (2048, 1080),
                                'Playground': (1920, 1080),
                                'Train': (1920, 1080)}
            
        elif self.split == 'advanced':
            self.scans = ['Auditorium', 'Ballroom', 'Courtroom',
                          'Museum', 'Palace', 'Temple']
            self.image_sizes = {'Auditorium': (1920, 1080),
                                'Ballroom': (1920, 1080),
                                'Courtroom': (1920, 1080),
                                'Museum': (1920, 1080),
                                'Palace': (1920, 1080),
                                'Temple': (1920, 1080)}
            
        for scan in self.scans:
            with open(os.path.join(self.datapath, self.split, scan, 'pair.txt')) as f:
                num_viewpoint = int(f.readline())
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    if len(src_views) != 0:
                        # self.metas += [(scan, -1, ref_view, src_views)]
                        self.metas += [(scan, ref_view, src_views, scan)]

        print("split: ", self.split, "metas:", len(self.metas))

    def define_transforms(self):
        self.transform_seg = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
   
    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = [line.rstrip() for line in f.readlines()]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
        extrinsics = extrinsics.reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
        intrinsics = intrinsics.reshape((3, 3))
        intrinsics[:2, :] /= 4.0
        
        depth_min = float(lines[11].split()[0])
        depth_max = float(lines[11].split()[1])

        return intrinsics, extrinsics, depth_min, depth_max

    def read_img(self, filename):
        img = Image.open(filename)
        # scale 0~255 to 0~1
        np_img = np.array(img, dtype=np.float32) / 255.

        return np_img


    def read_img_seg(self, filename):
        img = Image.open(filename)
        return self.transform_seg(img)


    def read_img_aug(self, filename):
        img = Image.open(filename)
        img = self.transform_aug(img)
        return img


    def read_depth(self, filename):
        # read pfm depth file
        return np.array(read_pfm(filename)[0], dtype=np.float32)


    def scale_mvs_input(self, img, intrinsics, max_w, max_h, base=32):
        h, w = img.shape[:2]
        # if h > max_h or w > max_w:
        #     scale = 1.0 * max_h / h
        #     if scale * w > max_w:
        #         scale = 1.0 * max_w / w
        #     new_w, new_h = scale * w // base * base, scale * h // base * base
        # else:
        #     new_w, new_h = 1.0 * w // base * base, 1.0 * h // base * base
        new_h, new_w = max_h, max_w

        scale_w = 1.0 * new_w / w
        scale_h = 1.0 * new_h / h
        intrinsics[0, :] *= scale_w
        intrinsics[1, :] *= scale_h

        img = cv2.resize(img, (int(new_w), int(new_h)))

        return img, intrinsics

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        global s_h, s_w
        # scan, _, ref_view, src_views = self.metas[idx]
        scan, ref_view, src_views, scene_name = self.metas[idx]
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.nviews-1]
        img_w, img_h = self.image_sizes[scan]

        imgs = []
        depth_values = None
        proj_matrices = []

        for i, vid in enumerate(view_ids):
            img_filename = os.path.join(self.datapath, self.split, scan, f'images/{vid:08d}.jpg')
            proj_mat_filename = os.path.join(self.datapath, self.split, scan, f'cams_1/{vid:08d}_cam.txt')

            img = self.read_img(img_filename)
            # img = self.read_img_seg(img_filename)
            intrinsics, extrinsics, depth_min_, depth_max_ = self.read_cam_file(proj_mat_filename)
            # scale input
            # img, intrinsics = self.scale_mvs_input(img, intrinsics, self.max_w, self.max_h)
            img, intrinsics = self.scale_mvs_input(img, intrinsics, self.img_wh[0], self.img_wh[1])
            img = self.transform_seg(img)

            imgs.append(img)
            # extrinsics, intrinsics
            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)  #
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics
            proj_matrices.append(proj_mat)

            if i == 0:  # reference view
                depth_min =  depth_min_
                depth_max = depth_max_
                depth_interval = (depth_max - depth_min) / (self.ndepths - 1)
                depth_values = np.arange(depth_min, depth_interval * (self.ndepths - 0.5) + depth_min, depth_interval,
                                         dtype=np.float32)

        #all
        # imgs = np.stack(imgs).transpose([0, 3, 1, 2])
        imgs = np.stack(imgs)
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

        return {"imgs": imgs,
                "proj_matrices": proj_matrices_ms,
                "depth_values": depth_values,
                "filename": scan + '/{}/' + '{:0>8}'.format(view_ids[0]) + "{}"}