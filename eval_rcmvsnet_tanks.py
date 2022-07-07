import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import sys
import cv2
from plyfile import PlyData, PlyElement
from PIL import Image
import math
import matplotlib as mpl
import matplotlib.cm as cm

from datasets import find_dataset_def
from models import *
from utils import *
from datasets.data_io import read_pfm, save_pfm
from models.casmvsnet import CascadeMVSNet_eval

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Predict depth, filter, and fuse')
parser.add_argument('--model', default='mvsnet', help='select model')

parser.add_argument('--dataset', default='tanks', help='select dataset')
parser.add_argument('--testpath', default='/home/dichang/datasets/TankandTemples',help='testing data dir for some scenes')
parser.add_argument('--split', default='intermediate', help='select data')

parser.add_argument('--batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--numdepth', type=int, default=192, help='the number of depth values')


parser.add_argument('--loadckpt', default='./pretrain/model_000014_cas.ckpt', help='load prestrained model of the rc-mvsnet')
parser.add_argument('--outdir', default='./tanks_exp', help='output dir of rc-mvsnet on TankandTemples')
parser.add_argument('--plydir', default='./tanks_submission', help='output dir of fusion points')

parser.add_argument('--display', action='store_true', help='display depth images and masks')

parser.add_argument('--share_cr', action='store_true', help='whether share the cost volume regularization')

parser.add_argument('--ndepths', type=str, default="48,32,8", help='ndepths')
parser.add_argument('--depth_inter_r', type=str, default="4,2,1", help='depth_intervals_ratio')
parser.add_argument('--cr_base_chs', type=str, default="8,8,8", help='cost regularization base channels')
parser.add_argument('--grad_method', type=str, default="detach", choices=["detach", "undetach"], help='grad method')


parser.add_argument('--num_view', type=int, default=7, help='num of view')
parser.add_argument('--max_h', type=int, default=1056, help='testing max h')
parser.add_argument('--max_w', type=int, default=1920, help='testing max w')
parser.add_argument('--fix_res', action='store_true', help='scene all using same res')


parser.add_argument('--num_worker', type=int, default=4, help='depth_filer worker')
parser.add_argument('--save_freq', type=int, default=20, help='save freq of local pcd')


#filter
parser.add_argument('--conf', type=float, default=0.9, help='prob confidence')
parser.add_argument('--thres_view', type=int, default=5, help='threshold of num view')

parser.add_argument('--geo_pixel_thres', type=float, default=1, help='pixel threshold for geometric consistency filtering')
parser.add_argument('--geo_depth_thres', type=float, default=0.01, help='depth threshold for geometric consistency filtering')

#filter by gimupa
parser.add_argument('--fusibile_exe_path', type=str, default='./fusion/fusibile/build/fusibile')
parser.add_argument('--prob_threshold', type=float, default='0.9')
parser.add_argument('--disp_threshold', type=float, default='0.25')
parser.add_argument('--num_consistent', type=float, default='4')

# model setting
parser.add_argument('--view_weight', type=str, default=False, help='PVSNet')
parser.add_argument('--PMVSNet', type=bool, default=False, help="use PMVSNet 7*1*1 + 1*3*3 convolution")
parser.add_argument('--dcn', type=bool, default=False, help="use deformable convolution in 2D backbone")
parser.add_argument('--gn', default=False,help='apply group normalization.')


parser.add_argument('--using_apex', action='store_true', help='using apex, need to install apex')
parser.add_argument('--sync_bn', action='store_true',help='enabling apex sync BN.')


# parse arguments and check
args = parser.parse_args()
print("argv:", sys.argv[1:])
print_args(args)
# os.environ["CUDA_VISIBLE_DEVICES"] = args.true_gpu

# read an image
def read_img(filename, img_wh):
    img = Image.open(filename)
    # scale 0~255 to 0~1
    np_img = np.array(img, dtype=np.float32) / 255.
    np_img = cv2.resize(np_img, img_wh, interpolation=cv2.INTER_LINEAR)
    
    return np_img

def read_cam_file(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
    extrinsics = extrinsics.reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
    intrinsics = intrinsics.reshape((3, 3))
    
    return intrinsics, extrinsics

# save a binary mask
def save_mask(filename, mask):
    assert mask.dtype == np.bool
    mask = mask.astype(np.uint8) * 255
    Image.fromarray(mask).save(filename)

def save_depth_img(filename, depth):
    # assert mask.dtype == np.bool
    depth = depth * 255
    depth = depth.astype(np.uint8)
    Image.fromarray(depth).save(filename)


def read_pair_file(filename):
    data = []
    with open(filename) as f:
        num_viewpoint = int(f.readline())
        # 49 viewpoints
        for view_idx in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            if len(src_views) != 0:
                data.append((ref_view, src_views))
    return data


def write_depth_img_2(filename, depth):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    vmax = np.percentile(depth, 95)
    normalizer = mpl.colors.Normalize(vmin=depth.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma_r')
    colormapped_im = (mapper.to_rgba(depth)[:, :, :3] * 255).astype(np.uint8)
    im = Image.fromarray(colormapped_im)
    im.save(filename)


# run MVS model to save depth maps
def save_depth(img_wh=(1920, 1056)):
    # dataset, dataloader
    MVSDataset = find_dataset_def(args.dataset)
    test_dataset = MVSDataset(args.testpath, args.split, args.num_view, img_wh, args.numdepth)
    TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=4, drop_last=False)

    model = CascadeMVSNet_eval(refine=False, ndepths=[int(nd) for nd in args.ndepths.split(",") if nd],
            depth_interals_ratio=[float(d_i) for d_i in args.depth_inter_r.split(",") if d_i],
            share_cr=args.share_cr,
            cr_base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch],
            grad_method=args.grad_method,norm=args.gn,dcn=args.dcn)

    # load checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict['model'], strict=True)
    model = nn.DataParallel(model)
    model.cuda()
    model.eval()
    
    with torch.no_grad():
        for batch_idx, sample in enumerate(TestImgLoader):
            start_time = time.time()
            sample_cuda = tocuda(sample)
            # outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_min"], sample_cuda["depth_max"])
            outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])

            outputs = tensor2numpy(outputs)
            del sample_cuda
            print('Iter {}/{}, time = {:.3f}'.format(batch_idx, len(TestImgLoader), time.time() - start_time))
            filenames = sample["filename"]

            # save depth maps and confidence maps
            for filename, depth_est, photometric_confidence in zip(filenames, outputs["depth"], outputs["photometric_confidence"]):
                depth_filename = os.path.join(args.outdir, filename.format('depth_est', '.pfm'))
                confidence_filename = os.path.join(args.outdir, filename.format('confidence', '.pfm'))
                os.makedirs(depth_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(confidence_filename.rsplit('/', 1)[0], exist_ok=True)
                # save depth maps
                # print(depth_est.shape)
                # depth_est = np.squeeze(depth_est, 0)
                save_pfm(depth_filename, depth_est)
                # save confidence maps
                save_pfm(confidence_filename, photometric_confidence)
                write_depth_img_2(depth_filename + ".png", depth_est)


# project the reference point cloud into the source view, then project back
def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    ## step1. project reference pixels to the source view
    # reference view x, y
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    # reference 3D space
    xyz_ref = np.matmul(np.linalg.inv(intrinsics_ref),
                        np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1]))
    # source 3D space
    xyz_src = np.matmul(np.matmul(extrinsics_src, np.linalg.inv(extrinsics_ref)),
                        np.vstack((xyz_ref, np.ones_like(x_ref))))[:3]
    # source view x, y
    K_xyz_src = np.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]

    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[0].reshape([height, width]).astype(np.float32)
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)
    sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR)
    # mask = sampled_depth_src > 0

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = np.matmul(np.linalg.inv(intrinsics_src),
                        np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1]))
    # reference 3D space
    xyz_reprojected = np.matmul(np.matmul(extrinsics_ref, np.linalg.inv(extrinsics_src)),
                                np.vstack((xyz_src, np.ones_like(x_ref))))[:3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)
    K_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src


def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src,
                                geo_pixel_thres=1, geo_depth_thres=0.01, dynamic_consistency_check=False):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref,
                                                     depth_src, intrinsics_src, extrinsics_src)
    # print(depth_ref.shape)
    # print(depth_reprojected.shape)
    # check |p_reproj-p_1| < 1
    dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    # check |d_reproj-d_1| / d_1 < 0.01
    # depth_ref = np.squeeze(depth_ref, 2)
    depth_diff = np.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref
    if dynamic_consistency_check:
        return dist, relative_depth_diff, depth_reprojected
    
    mask = np.logical_and(dist < geo_pixel_thres, relative_depth_diff < geo_depth_thres)
    
    depth_reprojected[~mask] = 0
    
    return mask, depth_reprojected, x2d_src, y2d_src


def filter_depth(scan_folder, out_folder, plyfilename, geo_pixel_thres, geo_depth_thres, photo_thres, img_wh, image_sizes, geo_mask_thres, n_views, scan):
    # the pair file
    pair_file = os.path.join(scan_folder, "pair.txt")
    # for the final point cloud
    vertexs = []
    vertex_colors = []

    pair_data = read_pair_file(pair_file)
    nviews = len(pair_data)
    original_w, original_h = image_sizes
    
    # for each reference view and the corresponding source views
    idx = 0
    for ref_view, src_views in pair_data:
        idx += 1
        print('[{}/{}] Processing scan: {}'.format(idx, len(pair_data), scan))
        # load the camera parameters
        ref_intrinsics, ref_extrinsics = read_cam_file(
            os.path.join(scan_folder, 'cams_1/{:0>8}_cam.txt'.format(ref_view)))
        ref_intrinsics[0] *= img_wh[0]/original_w
        ref_intrinsics[1] *= img_wh[1]/original_h
        # load the reference image
        ref_img = read_img(os.path.join(scan_folder, 'images/{:0>8}.jpg'.format(ref_view)),img_wh)
        
        # load the estimated depth of the reference view
        ref_depth_est = read_pfm(os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(ref_view)))[0]

        # load the photometric mask of the reference view
        confidence = read_pfm(os.path.join(out_folder, 'confidence/{:0>8}.pfm'.format(ref_view)))[0]
        
        photo_mask = confidence > photo_thres # prob_threshold

        all_srcview_depth_ests = []
        

        # compute the geometric mask
        geo_mask_sum = 0
        n_views = len(src_views)+1
        for i in range(n_views-1):
            src_view = src_views[i]
            
            src_intrinsics, src_extrinsics = read_cam_file(
                os.path.join(scan_folder, 'cams_1/{:0>8}_cam.txt'.format(src_view)))
            src_intrinsics[0] *= img_wh[0]/original_w
            src_intrinsics[1] *= img_wh[1]/original_h
            
            
            # the estimated depth of the source view
            src_depth_est = read_pfm(os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(src_view)))[0]

            geo_mask, depth_reprojected, _, _ = check_geometric_consistency(ref_depth_est, ref_intrinsics, ref_extrinsics,
                                                                      src_depth_est,
                                                                      src_intrinsics, src_extrinsics,
                                                                      geo_pixel_thres, geo_depth_thres)
            geo_mask_sum += geo_mask.astype(np.int32)
            all_srcview_depth_ests.append(depth_reprojected)
            

        depth_est_averaged = (sum(all_srcview_depth_ests) + ref_depth_est) / (geo_mask_sum + 1)
        
        geo_mask = geo_mask_sum >= geo_mask_thres # num_consistency
        final_mask = np.logical_and(photo_mask, geo_mask)
        

        os.makedirs(os.path.join(out_folder, "mask"), exist_ok=True)
        save_mask(os.path.join(out_folder, "mask/{:0>8}_photo.png".format(ref_view)), photo_mask)
        save_mask(os.path.join(out_folder, "mask/{:0>8}_geo.png".format(ref_view)), geo_mask)
        save_mask(os.path.join(out_folder, "mask/{:0>8}_final.png".format(ref_view)), final_mask)
        

        print("processing {}, ref-view{:0>2}, geo_mask:{:3f} photo_mask:{:3f} final_mask: {:3f}".format(scan_folder, ref_view,
                                                                geo_mask.mean(), photo_mask.mean(), final_mask.mean()))

        if args.display:
            import cv2
            cv2.imshow('ref_img', ref_img[:, :, ::-1])
            cv2.imshow('ref_depth', ref_depth_est )
            cv2.imshow('ref_depth * photo_mask', ref_depth_est * photo_mask.astype(np.float32))
            cv2.imshow('ref_depth * geo_mask', ref_depth_est * geo_mask.astype(np.float32) )
            cv2.imshow('ref_depth * mask', ref_depth_est * final_mask.astype(np.float32))
            cv2.waitKey(1)

        height, width = depth_est_averaged.shape[:2]
        x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))
        
        valid_points = final_mask
        # print("valid_points", valid_points.mean())
        x, y, depth = x[valid_points], y[valid_points], depth_est_averaged[valid_points]
        
        color = ref_img[valid_points]
        xyz_ref = np.matmul(    np.linalg.inv(ref_intrinsics),
                            np.vstack((x, y, np.ones_like(x))) * depth)
        xyz_world = np.matmul(np.linalg.inv(ref_extrinsics),
                              np.vstack((xyz_ref, np.ones_like(x))))[:3]
        vertexs.append(xyz_world.transpose((1, 0)))
        vertex_colors.append((color * 255).astype(np.uint8))


    vertexs = np.concatenate(vertexs, axis=0)
    vertex_colors = np.concatenate(vertex_colors, axis=0)
    vertexs = np.array([tuple(v) for v in vertexs], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_colors = np.array([tuple(v) for v in vertex_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]

    el = PlyElement.describe(vertex_all, 'vertex')
    PlyData([el]).write(plyfilename)
    print("saving the final model to", plyfilename)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    img_wh=(1920, 1056)
    # img_wh=(2048, 1184)

    # step1. save all the depth maps and the masks in outputs directory
    save_depth(img_wh)
    
    
    # intermediate dataset
    if args.split == "intermediate":

        scans = ['Family', 'Francis', 'Horse', 'Lighthouse',
                'M60', 'Panther', 'Playground', 'Train']
        
        image_sizes = {'Family': (1920, 1080),
                            'Francis': (1920, 1080),
                            'Horse': (1920, 1080),
                            'Lighthouse': (2048, 1080),
                            'M60': (2048, 1080),
                            'Panther': (2048, 1080),
                            'Playground': (1920, 1080),
                            'Train': (1920, 1080)}
        geo_mask_thres = {'Family': 6,
                            'Francis': 8,
                            'Horse': 4,
                            'Lighthouse': 7,
                            'M60': 6,
                            'Panther': 7,
                            'Playground': 7,
                            'Train': 6} # num_consistency
        photo_thres = {'Family': 0.9,
                            'Francis': 0.8,
                            'Horse': 0.8,
                            'Lighthouse': 0.8,
                            'M60': 0.9,
                            'Panther': 0.9,
                            'Playground': 0.85,
                            'Train': 0.9} # prob_threshold
        geo_pixel_thres = {'Family': 0.75,
                            'Francis': 1.0,
                            'Horse': 1.25,
                            'Lighthouse': 1.0,
                            'M60': 0.75,
                            'Panther': 1.0,
                            'Playground': 1.0,
                            'Train': 1.5}    # img_dist_thresh
        
        geo_depth_thres = {'Family': 0.01,
                            'Francis': 0.01,
                            'Horse': 0.01,
                            'Lighthouse': 0.01,
                            'M60': 0.005,
                            'Panther': 0.01,
                            'Playground': 0.01,
                            'Train': 0.01}   # depth_thresh
        for scan in scans:
            
            scan_folder = os.path.join(args.testpath, args.split, scan)
            out_folder = os.path.join(args.outdir, scan)
            # step2. filter saved depth maps with geometric constraints

            scan_ply_path = os.path.join(args.plydir, scan + '.ply')
            if os.path.exists(scan_ply_path):
                print('{} exists. skipped.'.format(scan_ply_path))
                continue
            
            filter_depth(scan_folder, out_folder, scan_ply_path, 
                geo_pixel_thres[scan], geo_depth_thres[scan], photo_thres[scan], img_wh, image_sizes[scan], geo_mask_thres[scan], args.num_view, scan) 
    # advanced dataset
    elif args.split == "advanced":

        scans = ['Auditorium', 'Ballroom', 'Courtroom',
                'Museum', 'Palace', 'Temple']
        
        image_sizes = {'Auditorium': (1920, 1080),
                                'Ballroom': (1920, 1080),
                                'Courtroom': (1920, 1080),
                                'Museum': (1920, 1080),
                                'Palace': (1920, 1080),
                                'Temple': (1920, 1080)}
        geo_mask_thres = {'Auditorium': 3,
                            'Ballroom': 4,
                            'Courtroom': 3,
                            'Museum': 4,
                            'Palace': 5,
                            'Temple': 3}

        photo_thres = {'Auditorium': 0.7,
                            'Ballroom': 0.8,
                            'Courtroom': 0.8,
                            'Museum': 0.8,
                            'Palace': 0.9,
                            'Temple': 0.8}
        geo_pixel_thres = {'Auditorium': 4.0,
                            'Ballroom': 4.0,
                            'Courtroom': 3.0,
                            'Museum': 4.0,
                            'Palace': 4.0,
                            'Temple': 4.0}    # img_dist_thresh
        
        geo_depth_thres = {'Auditorium': 0.005,
                            'Ballroom': 0.005,
                            'Courtroom': 0.005,
                            'Museum': 0.01,
                            'Palace': 0.005,
                            'Temple': 0.01}   # depth_thresh
        for scan in scans:
            
            scan_folder = os.path.join(args.testpath, args.split, scan)
            out_folder = os.path.join(args.outdir, scan)

            scan_ply_path = os.path.join(args.plydir, scan + '.ply')
            if os.path.exists(scan_ply_path):
                print('{} exists. skipped.'.format(scan_ply_path))
                continue

            # step2. filter saved depth maps with geometric constraints
            filter_depth(scan_folder, out_folder, scan_ply_path, 
                geo_pixel_thres[scan], geo_depth_thres[scan], photo_thres[scan], img_wh, image_sizes[scan], geo_mask_thres[scan], args.num_view, scan) 