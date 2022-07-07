
import argparse, os, sys, time, gc, datetime

from torch.functional import norm
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# import flow_vis

from datasets import find_dataset_def
from models import *
from models.casmvsnet import CascadeMVSNet
from models.render_consist_net import Rendering_Consistency_Net
from utils import *
from losses.unsup_loss import UnsupLossMultiStage
from losses.aug_loss import random_image_mask, AugLossMultiStage
from losses.sl1loss import SL1Loss

cudnn.benchmark = True

# arguments
parser = argparse.ArgumentParser(description='A PyTorch Implementation of RC-MVSNet')
parser.add_argument('--mode', default='train', help='train or test', choices=['train', 'test'])
parser.add_argument('--model', default='mvsnet', help='select model')
parser.add_argument('--device', default='cuda', help='select model')

parser.add_argument('--dataset', default='dtu_train', help='select dataset')
parser.add_argument('--trainpath', default="/cluster/51/dichang/datasets/mvsnet",help='train datapath')
parser.add_argument('--testpath', default="/cluster/51/dichang/datasets/mvsnet",help='test datapath')
parser.add_argument('--trainlist', default="./lists/dtu/train.txt",help='train list')
parser.add_argument('--testlist', default='./lists/dtu/test.txt',help='test list')

parser.add_argument('--epochs', type=int, default=15, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--lrepochs', type=str, default="10,12,14:2", help='epoch ids to downscale lr and the downscale rate')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')

parser.add_argument('--batch_size', type=int, default=1, help='train batch size')
parser.add_argument('--numdepth', type=int, default=192, help='the number of depth values')
parser.add_argument('--interval_scale', type=float, default=1.06, help='the number of depth values')
parser.add_argument('--num_view', type=int, default=3, help='the number of source views')

parser.add_argument('--logdir', default='./rc-mvsnet', help='the directory to save checkpoints/logs')
parser.add_argument('--resume', action='store_true', help='continue to train the model')

parser.add_argument('--summary_freq', type=int, default=10, help='print and summary frequency')
parser.add_argument('--save_freq', type=int, default=1, help='save checkpoint frequency')
parser.add_argument('--eval_freq', type=int, default=1, help='eval freq')

parser.add_argument('--random_seed', type=int, default=1, metavar='S', help='random seed')


parser.add_argument('--share_cr', action='store_true', help='whether share the cost volume regularization')
parser.add_argument('--ndepths', type=str, default="48,32,8", help='ndepths')
parser.add_argument('--depth_inter_r', type=str, default="4,2,1", help='depth_intervals_ratio')
parser.add_argument('--dlossw', type=str, default="0.5,1.0,2.0", help='depth loss weight for different stage')
parser.add_argument('--cr_base_chs', type=str, default="8,8,8", help='cost regularization base channels')
parser.add_argument('--grad_method', type=str, default="detach", choices=["detach", "undetach"], help='grad method')
parser.add_argument('--w_aug', type=float, default=0.01, help='weight of aug loss')


parser.add_argument('--true_gpu',default="0",help='using true gpu')
parser.add_argument('--gpu',default=[0],help='gpu')
#parser.add_argument('--master_port',default='10005',help='port number')
parser.add_argument('--master_port',default='11026',help='port number')

#   added for rendering network#################################################################################

parser.add_argument('--imgScale_train', type=float, default=1.0)
parser.add_argument('--imgScale_test', type=float, default=1.0)
parser.add_argument('--img_downscale', type=float, default=1.0)
parser.add_argument('--pad', type=int, default=0)



parser.add_argument('--use_color_volume', default=False, action="store_true",
                        help='project colors into a volume without indexing from image everytime')


parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
parser.add_argument("--pts_dim", type=int, default=3)
parser.add_argument("--dir_dim", type=int, default=3)


# training options
parser.add_argument("--netdepth", type=int, default=6,
                        help='layers in network')
parser.add_argument("--netwidth", type=int, default=128,
                        help='channels per layer')
parser.add_argument('--net_type', type=str, default='v0')



parser.add_argument("--netchunk", type=int, default=1024,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
parser.add_argument("--ckpt", type = str , default=None, 
                        help='specific weights npy file to reload for coarse network')
parser.add_argument("--N_samples", type=int, default=128,
                    help='number of coarse samples per ray')

parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
parser.add_argument('--use_disp', default=False, action="store_true",
                    help='use disparity depth sampling')
parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
## blender flags
parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

args = parser.parse_args()
    
num_gpus = len(args.gpu)
is_distributed = num_gpus > 1


# main function
def train(model, model_nerf,model_loss, aug_loss, test_model_loss, optimizer, TrainImgLoader, TestImgLoader, start_epoch, args):
    milestones = [len(TrainImgLoader) * int(epoch_idx) for epoch_idx in args.lrepochs.split(':')[0].split(',')]
    lr_gamma = 1 / float(args.lrepochs.split(':')[1])
    lr_scheduler = WarmupMultiStepLR(optimizer, milestones, gamma=lr_gamma, warmup_factor=1.0/3, warmup_iters=500,
                                                        last_epoch=len(TrainImgLoader) * start_epoch - 1)
    
    logger = SummaryWriter(args.logdir)

    for epoch_idx in range(start_epoch, args.epochs):
        print('Epoch {}:'.format(epoch_idx))
        global_step = len(TrainImgLoader) * epoch_idx
        avg_train_scalars = DictAverageMeter()
        avg_aug_scalars = DictAverageMeter()
        avg_nerf_scalars = DictAverageMeter()

        for batch_idx, sample in enumerate(TrainImgLoader):
            start_time = time.time()
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0
            # stage 1: standard self-supervision loss update
            loss, scalar_outputs, image_outputs, pseudo_depth,volume_feature,loss_base = train_sample(model, model_loss, optimizer, sample, args)
            if (not is_distributed) or (dist.get_rank() == 0):
                if do_summary:
                    save_scalars(logger, 'train', scalar_outputs, global_step)
                    save_images(logger, 'train', image_outputs, global_step)
                    print(
                       "Epoch {}/{}, Iter-S1 {}/{}, lr {:.6f}, train loss = {:.3f},  depth loss = {:.3f}, thres2mm_error = {:.3f}, thres2mm_accu = {:.3f}, thres4mm_error = {:.3f}, thres4mm_accu = {:.3f}, thres8mm_error = {:.3f}, thres8mm_accu = {:.3f},time = {:.3f}".format(
                           epoch_idx, args.epochs, batch_idx, len(TrainImgLoader),
                           optimizer.param_groups[0]["lr"], loss,
                           scalar_outputs['depth_loss_stage3'],
                           scalar_outputs['thres2mm_error'],scalar_outputs['thres2mm_accu'],
                           scalar_outputs['thres4mm_error'],scalar_outputs['thres4mm_accu'],
                           scalar_outputs['thres8mm_error'],scalar_outputs['thres8mm_accu'],
                           time.time() - start_time))
                avg_train_scalars.update(scalar_outputs)
                del scalar_outputs, image_outputs

            # stage 2: augmentation self-supervision loss update
            loss_t, aug_scalar_outputs, aug_image_outputs,loss_aug = train_sample_aug(model, aug_loss, optimizer, sample, args, pseudo_depth, epoch_idx)
            if (not is_distributed) or (dist.get_rank() == 0):
                if do_summary:
                    save_scalars(logger, 'train', aug_scalar_outputs, global_step)
                    save_images(logger, 'train', aug_image_outputs, global_step)
                    print(
                       "Epoch {}/{}, Iter-S2 {}/{}, lr {:.6f}, aug loss = {:.3f}, depth loss = {:.3f}, thres2mm_error = {:.3f}, thres2mm_accu = {:.3f}, thres4mm_error = {:.3f}, thres4mm_accu = {:.3f}, thres8mm_error = {:.3f}, thres8mm_accu = {:.3f}, time = {:.3f}".format(
                           epoch_idx, args.epochs, batch_idx, len(TrainImgLoader),
                           optimizer.param_groups[0]["lr"], loss_t,
                           aug_scalar_outputs['aug_loss_stage3'],
                           aug_scalar_outputs['thres2mm_error'],aug_scalar_outputs['thres2mm_accu'],
                           aug_scalar_outputs['thres4mm_error'],aug_scalar_outputs['thres4mm_accu'],
                           aug_scalar_outputs['thres8mm_error'],aug_scalar_outputs['thres8mm_accu'],
                           time.time() - start_time))
                avg_aug_scalars.update(aug_scalar_outputs)
                del aug_scalar_outputs, aug_image_outputs




           
            loss_nerf, nerf_scalar_outputs, nerf_image_outputs = train_render_net(volume_feature,model_nerf,sample,optimizer,pseudo_depth,epoch_idx,args,loss_base,loss_aug)
            if (not is_distributed) or (dist.get_rank() == 0):
                if do_summary:
                    save_scalars(logger, 'train', nerf_scalar_outputs, global_step)
                    save_images(logger, 'train', nerf_image_outputs, global_step)
                    print(
                       "Epoch {}/{}, Iter-S2 {}/{}, lr {:.6f}, overall loss = {:.9f}, img loss = {:.9f},depth loss = {:.9f}, time = {:.3f}".format(
                           epoch_idx, args.epochs, batch_idx, len(TrainImgLoader),
                           optimizer.param_groups[0]["lr"], loss_nerf,
                           nerf_scalar_outputs['img_loss'], nerf_scalar_outputs['depth_loss'],
                           time.time() - start_time))
                avg_nerf_scalars.update(nerf_scalar_outputs)
                del nerf_scalar_outputs, nerf_image_outputs


            lr_scheduler.step()

        # checkpoint
        if (not is_distributed) or (dist.get_rank() == 0):
            save_scalars(logger, 'fulltrain', avg_train_scalars.mean(), global_step)
            save_scalars(logger, 'fulltrain', avg_aug_scalars.mean(), global_step)
            save_scalars(logger, 'fulltrain', avg_nerf_scalars.mean(), global_step)
            print("avg_train_scalars:", avg_train_scalars.mean())
            print("avg_aug_scalars:", avg_aug_scalars.mean())
            print("avg_nerf_scalars:", avg_nerf_scalars.mean())
            if (epoch_idx + 1) % args.save_freq == 0:
                torch.save({
                    'epoch': epoch_idx,
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict()},
                    "{}/model_{:0>6}_cas.ckpt".format(args.logdir, epoch_idx))
                torch.save({
                    # 'epoch': epoch_idx,
                    'model': model_nerf.module.state_dict(),
                    # 'optimizer': optimizer.state_dict()
                    },

                    "{}/model_{:0>6}_nerf.ckpt".format(args.logdir, epoch_idx))
        gc.collect()

        # testing
        if (epoch_idx % args.eval_freq == 0) or (epoch_idx == args.epochs - 1):
            avg_test_scalars = DictAverageMeter()
            for batch_idx, sample in enumerate(TestImgLoader):
                start_time = time.time()
                global_step = len(TrainImgLoader) * epoch_idx + batch_idx
                do_summary = global_step % args.summary_freq == 0
                loss, scalar_outputs, image_outputs = test_sample_depth(model, test_model_loss, sample, args)
                if (not is_distributed) or (dist.get_rank() == 0):
                    if do_summary:
                        save_scalars(logger, 'test', scalar_outputs, global_step)
                        save_images(logger, 'test', image_outputs, global_step)
                        print("Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, depth loss = {:.3f}, thres2mm_accu = {:.3f},thres4mm_accu = {:.3f},thres8mm_accu = {:.3f},thres2mm_error = {:.3f},thres4mm_error = {:.3f},thres8mm_error = {:.3f},time = {:3f}".format(
                                                                            epoch_idx, args.epochs,
                                                                            batch_idx,
                                                                            len(TestImgLoader), loss,
                                                                            scalar_outputs["depth_loss"],
                                                                            scalar_outputs["thres2mm_accu"],
                                                                            scalar_outputs["thres4mm_accu"],
                                                                            scalar_outputs["thres8mm_accu"],
                                                                            scalar_outputs["thres2mm_error"],
                                                                            scalar_outputs["thres4mm_error"],
                                                                            scalar_outputs["thres8mm_error"],
                                                                            time.time() - start_time))
                    avg_test_scalars.update(scalar_outputs)
                    del scalar_outputs, image_outputs

            if (not is_distributed) or (dist.get_rank() == 0):
                save_scalars(logger, 'fulltest', avg_test_scalars.mean(), global_step)
                print("avg_test_scalars:", avg_test_scalars.mean())
            # gc.collect()


def test(model, model_loss, TestImgLoader, args):
    avg_test_scalars = DictAverageMeter()
    for batch_idx, sample in enumerate(TestImgLoader):
        start_time = time.time()
        loss, scalar_outputs, image_outputs = test_sample_depth(model, model_loss, sample, args)
        avg_test_scalars.update(scalar_outputs)
        del scalar_outputs, image_outputs
        if (not is_distributed) or (dist.get_rank() == 0):
            print('Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(batch_idx, len(TestImgLoader), loss,
                                                                        time.time() - start_time))
            if batch_idx % 100 == 0:
                print("Iter {}/{}, test results = {}".format(batch_idx, len(TestImgLoader), avg_test_scalars.mean()))
    if (not is_distributed) or (dist.get_rank() == 0):
        print("final", avg_test_scalars.mean())


    
def train_render_net(volume_feature,model_nerf,sample,optimizer,pseudo_depth,epoch_idx,args,loss_base,loss_aug):
    model_nerf.train()
    # optimizer.zero_grad()
    
    loss = loss_base + loss_aug

    rgb, disp, acc, depth_pred, alpha, ret, rays_depth, target_s = model_nerf(volume_feature,pseudo_depth,sample)
    mask_dtu = tocuda(sample)["mask"]["stage3"]



    ##################  rendering #####################
    img_loss = img2mse(rgb, target_s)
    loss = loss + img_loss


    mask = rays_depth > 0
    sl1loss = SL1Loss()
    depth_loss = sl1loss(depth_pred, rays_depth, mask)
    loss += depth_loss
    abs_err = abs_error(depth_pred, rays_depth, mask).mean()


    psnr = mse2psnr(img2mse(rgb.cpu()[mask], target_s.cpu()[mask]))
    psnr_out = mse2psnr(img2mse(rgb.cpu()[~mask], target_s.cpu()[~mask]))

    # with torch.no_grad():
    #     print('train/loss', loss.item())
    #     print('train/img_mse_loss', img_loss.item())
    #     print('train/PSNR', psnr.item())


    loss.backward()
    optimizer.step()
    image_outputs = {"nerf_depth_est": pseudo_depth * mask_dtu,
                     "nerf_depth_est_nomask": pseudo_depth
                         }

    scalar_outputs = {  'overall_loss':loss,
                            'img_loss':img_loss,
                            'depth_loss':depth_loss,
                            'acc_l_2mm': acc_threshold(depth_pred, rays_depth, mask,2).mean(),
                            'acc_l_10mm': acc_threshold(depth_pred, rays_depth, mask,10).mean(),
                            'acc_l_20mm': acc_threshold(depth_pred, rays_depth, mask,20).mean(),
                            'abs_err':abs_err
        }
    return  tensor2float(scalar_outputs['overall_loss']),tensor2float(scalar_outputs),tensor2numpy(image_outputs),




def train_sample(model, model_loss, optimizer, sample, args):
    model.train()
    optimizer.zero_grad()

    sample_cuda = tocuda(sample)
    depth_gt_ms = sample_cuda["depth"]
    mask_ms = sample_cuda["mask"]

    num_stage = len([int(nd) for nd in args.ndepths.split(",") if nd])
    depth_gt = depth_gt_ms["stage{}".format(num_stage)]
    mask = mask_ms["stage{}".format(num_stage)]

    outputs,volume_feature = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
    depth_est = outputs["depth"]
    
    repr_loss, scalars = model_loss(outputs, sample_cuda["center_imgs"], sample_cuda["proj_matrices"],
                               dlossw=[float(e) for e in args.dlossw.split(",") if e])

    loss = repr_loss

    
    scalar_outputs = {"loss": loss,
                      "repr_loss": repr_loss, 
                      # "depth_loss": depth_loss,
                      "abs_depth_error": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5),
                      "thres2mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 2),
                      "thres4mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 4),
                      "thres8mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 8),
                      "thres2mm_accu": 1.0 - Thres_metrics(depth_est, depth_gt, mask > 0.5, 2),
                      "thres4mm_accu": 1.0 - Thres_metrics(depth_est, depth_gt, mask > 0.5, 4),
                      "thres8mm_accu": 1.0 - Thres_metrics(depth_est, depth_gt, mask > 0.5, 8),}

    for key in scalars.keys():
        scalar_outputs[key] = scalars[key]

    image_outputs = {"depth_est": depth_est * mask,
                     "depth_est_nomask": depth_est,
                     "depth_gt": sample["depth"]["stage1"],
                     "ref_img": sample["imgs"][:, 0],
                     "mask": sample["mask"]["stage1"],
                     "errormap": (depth_est - depth_gt).abs() * mask,
                     }

    if is_distributed:
        scalar_outputs = reduce_scalar_outputs(scalar_outputs)

    return tensor2float(scalar_outputs["loss"]), tensor2float(scalar_outputs), tensor2numpy(image_outputs), depth_est.detach(),volume_feature,loss


def adjust_w_aug(epoch_idx, w_aug):
    if epoch_idx >= 2 - 1:
        w_aug *= 2
    if epoch_idx >= 4 - 1:
        w_aug *= 2
    if epoch_idx >= 6 - 1:
        w_aug *= 2
    if epoch_idx >= 8 - 1:
        w_aug *= 2
    if epoch_idx >= 10 - 1:
        w_aug *= 2
    # if epoch_idx >= 12 - 1:
    #     w_aug *= 2
    # if epoch_idx >= 14 - 1:
        # w_aug *= 2
    return w_aug


def train_sample_aug(model, aug_loss, optimizer, sample, args, pseudo_depth, epoch_idx):
    model.train()
    # optimizer.zero_grad()

    sample_cuda = tocuda(sample)
    depth_gt_ms = sample_cuda["depth"]
    mask_ms = sample_cuda["mask"]

    num_stage = len([int(nd) for nd in args.ndepths.split(",") if nd])
    depth_gt = depth_gt_ms["stage{}".format(num_stage)]
    mask = mask_ms["stage{}".format(num_stage)]

    # augmentation
    imgs_aug = sample_cuda["imgs_aug"]
    ref_img = imgs_aug[:, 0]
    ref_img, filter_mask = random_image_mask(ref_img, filter_size=(ref_img.size(2) // 3, ref_img.size(3) // 3))
    imgs_aug[:, 0] = ref_img

    outputs,_ = model(imgs_aug, sample_cuda["proj_matrices"], sample_cuda["depth_values"])
    depth_est = outputs["depth"]


    dlossw = [float(e) for e in args.dlossw.split(",") if e]
    loss, scalars = aug_loss(outputs, pseudo_depth, mask_ms, filter_mask, dlossw=dlossw)

    # adjust w_aug
    w_aug = adjust_w_aug(epoch_idx, args.w_aug)
    loss = loss * w_aug



    scalar_outputs = {"aug_loss": loss,
                      "thres2mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 2),
                      "thres4mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 4),
                      "thres8mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 8),
                      "thres2mm_accu": 1.0 - Thres_metrics(depth_est, depth_gt, mask > 0.5, 2),
                      "thres4mm_accu": 1.0 - Thres_metrics(depth_est, depth_gt, mask > 0.5, 4),
                      "thres8mm_accu": 1.0 - Thres_metrics(depth_est, depth_gt, mask > 0.5, 8)}
    for key in scalars.keys():
        scalar_outputs[key] = scalars[key]

    image_outputs = {"aug_depth_est": depth_est * mask,
                     "aug_depth_est_nomask": depth_est
                     }

    if is_distributed:
        scalar_outputs = reduce_scalar_outputs(scalar_outputs)

    return tensor2float(scalar_outputs["aug_loss"]), tensor2float(scalar_outputs), tensor2numpy(image_outputs),loss 



@make_nograd_func
def test_sample_depth(model, model_loss, sample, args):
    if is_distributed:
        model_eval = model.module
    else:
        model_eval = model
    model_eval.eval()

    sample_cuda = tocuda(sample)
    depth_gt_ms = sample_cuda["depth"]
    mask_ms = sample_cuda["mask"]

    num_stage = len([int(nd) for nd in args.ndepths.split(",") if nd])
    depth_gt = depth_gt_ms["stage{}".format(num_stage)]
    mask = mask_ms["stage{}".format(num_stage)]

    outputs,_ = model_eval(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
    depth_est = outputs["depth"]

    loss, depth_loss = model_loss(outputs, depth_gt_ms, mask_ms, dlossw=[float(e) for e in args.dlossw.split(",") if e])

    scalar_outputs = {"loss": loss,
                      "depth_loss": depth_loss,
                      "abs_depth_error": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5),
                      "thres2mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 2),
                      "thres4mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 4),
                      "thres8mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 8),
                    #   "thres14mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 14),
                    #   "thres20mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, 20),
                      "thres2mm_accu": 1- Thres_metrics(depth_est, depth_gt, mask > 0.5, 2),
                      "thres4mm_accu": 1 - Thres_metrics(depth_est, depth_gt, mask > 0.5, 4),
                      "thres8mm_accu": 1 - Thres_metrics(depth_est, depth_gt, mask > 0.5, 8),
                      "thres2mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [0, 2.0]),
                      "thres4mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [2.0, 4.0]),
                      "thres8mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [4.0, 8.0]),
                    #   "thres14mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [8.0, 14.0]),
                    #   "thres20mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [14.0, 20.0]),
                    #   "thres>20mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [20.0, 1e5]),
                    }

    image_outputs = {"depth_est": depth_est * mask,
                     "depth_est_nomask": depth_est,
                     "depth_gt": sample["depth"]["stage1"],
                     "ref_img": sample["imgs"][:, 0],
                     "mask": sample["mask"]["stage1"],
                     "errormap": (depth_est - depth_gt).abs() * mask}

    if is_distributed:
        scalar_outputs = reduce_scalar_outputs(scalar_outputs)

    return tensor2float(scalar_outputs["loss"]), tensor2float(scalar_outputs), tensor2numpy(image_outputs)


def train_begin(rank,args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.master_port
    torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=args.world_size)

    synchronize()
    set_random_seed(args.random_seed)
    torch.cuda.set_device(rank)

    # model, optimizer

    model = CascadeMVSNet(refine=False, ndepths=[int(nd) for nd in args.ndepths.split(",") if nd],
                            depth_interals_ratio=[float(d_i) for d_i in args.depth_inter_r.split(",") if d_i],
                            share_cr=args.share_cr,
                            cr_base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch],
                            grad_method=args.grad_method)

    # to device
    model.to(rank)

    model_nerf = Rendering_Consistency_Net(args)
    model_nerf.to(rank)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model_nerf = nn.SyncBatchNorm.convert_sync_batchnorm(model_nerf)
    print(model)
    print(model_nerf)
    model_loss = UnsupLossMultiStage().to(rank)

    aug_loss = AugLossMultiStage().to(rank)
    test_model_loss = cas_mvsnet_loss

    
 
    

    optimizer = optim.Adam(list(filter(lambda p: p.requires_grad, model.parameters())) + list(model_nerf.parameters()), lr=args.lr, betas=(0.9, 0.999),
                           weight_decay=args.wd)

    # load parameters
    start_epoch = 0
    if args.resume:
        saved_models = [fn for fn in os.listdir(args.logdir) if fn.endswith("cas.ckpt")]
        saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[1]))
        saved_models_nerf = [fn for fn in os.listdir(args.logdir) if fn.endswith("nerf.ckpt")]
        saved_models_nerf = sorted(saved_models_nerf, key=lambda x: int(x.split('_')[1]))
        # use the latest checkpoint file
        loadckpt = os.path.join(args.logdir, saved_models[-1])
        print("resuming", loadckpt)
        loadckpt_nerf = os.path.join(args.logdir, saved_models_nerf[-1])
        print("resuming nerf model", loadckpt_nerf)
        state_dict = torch.load(loadckpt, map_location=torch.device("cpu"))
        state_dict_nerf = torch.load(loadckpt_nerf, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict['model'])
        model_nerf.load_state_dict(state_dict_nerf['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        start_epoch = state_dict['epoch'] + 1



    print("start at epoch {}".format(start_epoch))
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


    if is_distributed:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[rank],find_unused_parameters=False
            # find_unused_parameters=False,
            # this should be removed if we update BatchNorm stats
            # broadcast_buffers=False,
        )
        model_nerf = torch.nn.parallel.DistributedDataParallel(
            model_nerf, device_ids=[rank],find_unused_parameters=False
            # find_unused_parameters=False,
            # this should be removed if we update BatchNorm stats
            # broadcast_buffers=False,
        )
    else:
        if torch.cuda.is_available():
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
            model_nerf = nn.DataParallel(model_nerf)

    # dataset, dataloader
    MVSDataset = find_dataset_def(args.dataset)
    # train_dataset = MVSDataset(args.trainpath, args.trainlist, "train", 3, args.numdepth, args.interval_scale)
    train_dataset = MVSDataset(args.trainpath, args.trainlist, "train", args.num_view+1, args.numdepth, args.interval_scale)
    test_MVSDataset = find_dataset_def('dtu_yao')
    test_dataset = test_MVSDataset(args.testpath, args.testlist, "test", 5, args.numdepth, args.interval_scale)

    if is_distributed:
        train_sampler = torch.utils.data.DistributedSampler(train_dataset, num_replicas=args.world_size,
                                                            rank=rank)
        test_sampler = torch.utils.data.DistributedSampler(test_dataset, num_replicas=args.world_size,
                                                           rank=rank)

        TrainImgLoader = DataLoader(train_dataset, args.batch_size, sampler=train_sampler, num_workers=1,
                                    drop_last=True)
        TestImgLoader = DataLoader(test_dataset, args.batch_size, sampler=test_sampler, num_workers=1, drop_last=False)
    else:
        TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=1, drop_last=True)
        TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=1, drop_last=False)

    if args.mode == "train":
        train(model,model_nerf, model_loss, aug_loss, test_model_loss, optimizer, TrainImgLoader, TestImgLoader, start_epoch, args)
    elif args.mode == "test":
        test(model, test_model_loss, TestImgLoader, args)
    else:
        raise NotImplementedError

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.true_gpu
    if args.resume:
        assert args.mode == "train"
        # assert args.loadckpt is None
    if args.testpath is None:
        args.testpath = args.trainpath
    world_size = num_gpus
    args.world_size = world_size

    if args.mode == "train":
        if not os.path.isdir(args.logdir):
            os.makedirs(args.logdir)
        current_time_str = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
        print("current time", current_time_str)
        print("creating new summary file")
        
    print("argv:", sys.argv[1:])
    print_args(args)
    # device = torch.device(args.device)
    import torch.multiprocessing as mp
    mp.spawn(train_begin,
        args=(args,),
        nprocs=world_size,
        join=True)

