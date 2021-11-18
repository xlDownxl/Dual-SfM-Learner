import argparse
import time
import csv
import datetime
from datasets.shifted_sequence_folder import StillBox
from path import Path

import numpy as np
import torch
torch.cuda.empty_cache()
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import random
import models
from inverse_warp import pose_vec2mat, inverse_rotate
import custom_transforms
from utils import tensor2array, save_checkpoint
from datasets.sequence_folders import SequenceFolder
from datasets.pair_folders import PairFolder
from loss_functions import compute_smooth_loss, compute_photo_and_geometry_loss, compute_errors, smooth_loss
from logger import TermLogger, AverageMeter
from tensorboardX import SummaryWriter
from collections import OrderedDict
import os
from utils import log_output_tensorboard
from utils import save_path_formatter
parser = argparse.ArgumentParser(description='Structure from Motion Learner training on KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--folder-type', type=str, choices=['sequence', 'pair'], default='sequence', help='the dataset dype to train')
parser.add_argument('--sequence-length', type=int, metavar='N', help='sequence length for training', default=3)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--epoch-size', default=0, type=int, metavar='N', help='manual epoch size (will match dataset size if not set)')
parser.add_argument('-b', '--batch-size', default=4, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M', help='beta parameters for adam')
parser.add_argument('--weight-decay', '--wd', default=0, type=float, metavar='W', help='weight decay')
parser.add_argument('--print-freq', default=10, type=int, metavar='N', help='print frequency')
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--log-summary', default='progress_log_summary.csv', metavar='PATH', help='csv where to save per-epoch train and valid stats')
parser.add_argument('--log-full', default='progress_log_full.csv', metavar='PATH', help='csv where to save per-gradient descent train stats')
parser.add_argument('--log-output', action='store_true', help='will log dispnet outputs at validation step')
parser.add_argument('--resnet-layers',  type=int, default=18, choices=[18, 50], help='number of ResNet layers for depth estimation.')
parser.add_argument('--num-scales', '--number-of-scales', type=int, help='the number of scales', metavar='W', default=1)
parser.add_argument('-p', '--photo-loss-weight', type=float, help='weight for photometric loss', metavar='W', default=1)
parser.add_argument('-s', '--smooth-loss-weight', type=float, help='weight for disparity smoothness loss', metavar='W', default=0.1)
parser.add_argument('-c', '--geometry-consistency-weight', type=float, help='weight for depth consistency loss', metavar='W', default=0.5)
parser.add_argument('--with-ssim', type=int, default=1, help='with ssim or not')
parser.add_argument('--with-mask', type=int, default=1, help='with the the mask for moving objects and occlusions or not')
parser.add_argument('--with-auto-mask', type=int,  default=0, help='with the the mask for stationary points')
parser.add_argument('--with-pretrain', type=int,  default=1, help='with or without imagenet pretrain for resnet')
parser.add_argument('--dataset', type=str, choices=['kitti', 'nyu-rectified','nyu-depth','tum', 'stillbox','crane'], default='kitti', help='the dataset to train')

parser.add_argument('--pretrained-disp', dest='pretrained_disp', default=None, metavar='PATH', help='path to pre-trained dispnet model')
parser.add_argument('--pretrained-pose', dest='pretrained_pose', default=None, metavar='PATH', help='path to pre-trained Pose net model')
parser.add_argument('--name', dest='name', type=str, required=True, help='name of the experiment, checkpoints are stored in checpoints/name')
parser.add_argument('--padding-mode', type=str, choices=['zeros', 'border'], default='zeros',
                    help='padding mode for image warping : this is important for photometric differenciation when going outside target image.'
                         ' zeros will null gradients outside target image.'
                         ' border will only null gradients of the coordinate outside (x or y)')
parser.add_argument('--with-gt', action='store_true', help='use ground truth for validation. \
                    You need to store it in npy 2D arrays see data/kitti_raw_loader.py for an example')
parser.add_argument('--width', type=int, help='use ground truth for validation. \
                    You need to store it in npy 2D arrays see data/kitti_raw_loader.py for an example')
parser.add_argument('--height', type=int, help='use ground truth for validation. \
                    You need to store it in npy 2D arrays see data/kitti_raw_loader.py for an example')
parser.add_argument('--fps', type=int, help='Changes the distance b etween target and sourc eimage. If higher than one, its not the neighbour image but a more distant one', default=1)
parser.add_argument('--downsample', type=int, help='', default=0)
parser.add_argument('--edge-smooth', type=int, help='', default=1)
parser.add_argument('--with-crane', action='store_true',)
parser.add_argument('--dont_eval', action='store_true',)

best_error = -1
n_iter = 0
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.autograd.set_detect_anomaly(True)


def main():
    def make_param_file(args,save_path):    
        args_dict = vars(args)
        data_folder_name = str(Path(args_dict['data']).normpath().name)
        folder_string = []
        folder_string.append('{} epochs'.format(args_dict['epochs']))
        keys_with_prefix = OrderedDict()
        keys_with_prefix['epoch_size'] = 'epoch_size '
        keys_with_prefix['sequence_length'] = 'sequence_length '
        keys_with_prefix['padding_mode'] = 'padding '
        keys_with_prefix['batch_size'] = 'batch_size '
        keys_with_prefix['lr'] = 'lr '
        keys_with_prefix['photo_loss_weight'] = 'photo_loss_weight '
        keys_with_prefix['smooth_loss_weight'] = 'smooth_loss_weight '
        keys_with_prefix['geometry_consistency_weight'] = 'geometry-consistency-weight '
        keys_with_prefix['width'] = 'width '
        keys_with_prefix['height'] = 'height '
        keys_with_prefix['with_gt'] = 'with_gt '
        keys_with_prefix['with_ssim'] = 'ssim weight '
        keys_with_prefix['resnet_layers'] = 'number of dispnet layers '   
        keys_with_prefix['with_pretrain'] = 'using pretrained dispnet '                  
        keys_with_prefix['with_mask'] = 'use geometric consistency mask '
        keys_with_prefix['with_auto_mask'] = 'use auto mask against static pixels '  
        keys_with_prefix['fps'] = 'distance between training frames ' 
        keys_with_prefix['downsample'] = 'downsampling factor of images ' 
        keys_with_prefix['dataset'] = 'dataset ' 
        keys_with_prefix['name'] = 'name ' 
        keys_with_prefix['edge_smooth'] = 'using edge smooth (instead of old smooth) ' 
        keys_with_prefix['pretrained_pose'] =' pretrained posenet '
        keys_with_prefix['pretrained_disp'] =' pretrained dispnet '

        for key, prefix in keys_with_prefix.items():
            value = args_dict[key]
            folder_string.append('{}{}'.format(prefix, value))

        timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
        folder_string.append('timestamp '+timestamp)
        commit_id = Path(os.popen("git log --pretty=format:'%h' -n 1").read())
        folder_string.append('Git Commit ID '+commit_id)
        commit_message = Path(os.popen("git log -1").read())
        folder_string.append('Git Message '+commit_message)
        params = '\n'.join(folder_string)
        with open(save_path/'params.txt', 'w') as f:
            f.write(params)
    global best_error, n_iter, device
    args = parser.parse_args()

    save_path = save_path_formatter(args, parser)
    args.save_path = 'checkpoints'/save_path
   
    print('=> will save everything to {}'.format(args.save_path))
    args.save_path.makedirs_p()
    make_param_file(args,args.save_path)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

    tb_writer = SummaryWriter(args.save_path)

    # Data loading code
    normalize = custom_transforms.Normalize(mean=[0.45, 0.45, 0.45],
                                            std=[0.225, 0.225, 0.225])

    train_transform = custom_transforms.Compose([
        custom_transforms.RandomHorizontalFlip(),
        custom_transforms.RandomScaleCrop(),
        custom_transforms.ArrayToTensor(),
        normalize
    ])

    valid_transform = custom_transforms.Compose([custom_transforms.ArrayToTensor(), normalize])

    print("=> fetching scenes in '{}'".format(args.data))
    if args.dataset=="stillbox":
        from datasets.shifted_sequence_folder import StillBox
        train_set = StillBox(
            args.data,
            transform=train_transform,
            seed=args.seed,
            train=True,
            with_depth_gt=False,
            with_pose_gt=False,
            sequence_length=args.sequence_length
        )
    else:
        if args.folder_type == 'sequence':
            train_set = SequenceFolder(
                args.data,
                transform=train_transform,
                skip_frames=args.fps,
                seed=args.seed,
                width=args.width,
                height=args.height,
                train=True,
                sequence_length=args.sequence_length,
                dataset=args.dataset,
                downsample=args.downsample,
            )
        else:
            train_set = PairFolder(
                args.data,
                seed=args.seed,
                train=True,
                transform=train_transform
            )


    # if no Groundtruth is avalaible, Validation set is the same type as training set to measure photometric loss from warping
    if args.dataset=="stillbox":
        from datasets.shifted_sequence_folder import StillBox
        val_set = StillBox(
            args.data,
            transform=valid_transform,
            seed=args.seed,
            train=False,
            with_depth_gt=True,
            with_pose_gt=False,
            sequence_length=args.sequence_length
        )
    else:
        
        if args.with_gt:
           
            val_set = SequenceFolder(
                args.data,
                transform=valid_transform,
                width=args.width,
                height=args.height,
                seed=args.seed,
                train=False,
                sequence_length=args.sequence_length,
                dataset=args.dataset,
                downsample=args.downsample,
                with_gt=True,
            )
        else:
            val_set = SequenceFolder(
                args.data,
                transform=valid_transform,
                width=args.width,
                height=args.height,
                seed=args.seed,
                train=False,
                sequence_length=args.sequence_length,
                dataset=args.dataset,
                downsample=args.downsample,
                with_gt=True,
            )
        
    print('{} samples found in {} train scenes'.format(len(train_set), len(train_set.scenes)))
    print('{} samples found in {} valid scenes'.format(len(val_set), len(val_set.scenes)))
    
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)


    if args.epoch_size == 0:
        args.epoch_size = len(train_loader)

    # create model
    print("=> creating model")
    disp_net = models.DispResNet(args.resnet_layers, args.with_pretrain).to(device)
    pose_net = models.PoseResNet(18, args.with_pretrain).to(device)

    # load parameters
    if args.pretrained_disp:
        print("=> using pre-trained weights for DispResNet")
        weights = torch.load(args.pretrained_disp)
        disp_net.load_state_dict(weights['state_dict'], strict=False)

    if args.pretrained_pose:
        print("=> using pre-trained weights for PoseResNet")
        weights = torch.load(args.pretrained_pose)
        pose_net.load_state_dict(weights['state_dict'], strict=False)

    disp_net = torch.nn.DataParallel(disp_net)
    pose_net = torch.nn.DataParallel(pose_net)

    print('=> setting adam solver')
    optim_params = [
        {'params': disp_net.parameters(), 'lr': args.lr},
        {'params': pose_net.parameters(), 'lr': args.lr}
    ]
    optimizer = torch.optim.Adam(optim_params,
                                 betas=(args.momentum, args.beta),
                                 weight_decay=args.weight_decay)

    with open(args.save_path/args.log_summary, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss', 'validation_loss'])

    with open(args.save_path/args.log_full, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss', 'photo_loss', 'smooth_loss', 'geometry_consistency_loss'])

   

    for epoch in range(args.epochs):
        print("Epoch {}".format(str(epoch)))

        train_loss = train(args, train_loader, disp_net, pose_net, optimizer, args.epoch_size,  tb_writer)

        # evaluate on validation set
        if not args.dont_eval:
            if args.with_gt:
                errors, error_names = validate_with_gt(args, val_loader, disp_net, pose_net, epoch,  tb_writer)
            else:
                errors, error_names = validate_without_gt(args, val_loader, disp_net, pose_net, epoch,  tb_writer)
            for error, name in zip(errors, error_names):
                tb_writer.add_scalar(name, error, epoch)


        # Up to you to chose the most relevant error to measure your model's performance, careful some measures are to maximize (such as a1,a2,a3)
            decisive_error = errors[1]
            if best_error < 0:
                best_error = decisive_error

        # remember lowest error and save checkpoint
            is_best = decisive_error < best_error
            best_error = min(best_error, decisive_error)
        else:
            is_best=False
        
        
        save_checkpoint(
            args.save_path, {
                'epoch': epoch + 1,
                'state_dict': disp_net.module.state_dict()
            }, {
                'epoch': epoch + 1,
                'state_dict': pose_net.module.state_dict()
            },
            is_best)

       

def train(args, train_loader, disp_net, pose_net, optimizer, epoch_size,  tb_writer):
    global n_iter, device
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(precision=4)
    w1, w2, w3 = args.photo_loss_weight, args.smooth_loss_weight, args.geometry_consistency_weight

    # switch to train mode
    disp_net.train()
    pose_net.train()

    end = time.time()

    for i, (tgt_img, ref_imgs, intrinsics, depth) in enumerate(train_loader):
        log_losses = i > 0 and n_iter % args.print_freq == 0

        # measure data loading time
        data_time.update(time.time() - end)
        tgt_img = tgt_img.to(device)
        ref_imgs = [img.to(device) for img in ref_imgs]
        intrinsics = intrinsics.to(device)
        
       
        poses, poses_inv = compute_pose_with_inv(pose_net, tgt_img, ref_imgs,)

        # compute output
        
        tgt_depth, ref_depths, _ = compute_depth(disp_net, tgt_img, ref_imgs, poses, poses_inv, intrinsics, args)

        loss_1, loss_3 = compute_photo_and_geometry_loss(tgt_img, ref_imgs, intrinsics, tgt_depth, ref_depths,
                                                         poses, poses_inv, args.num_scales, args.with_ssim,
                                                         args.with_mask, args.with_auto_mask, args.padding_mode)
        if args.edge_smooth:
            loss_2 = compute_smooth_loss(tgt_depth, tgt_img, ref_depths, ref_imgs)
        else:
            loss_2 = smooth_loss([tgt_depth]+ ref_depths)

        loss = w1*loss_1 + w2*loss_2 + w3*loss_3

        if log_losses:
            tb_writer.add_scalar('photometric_error', loss_1.item(), n_iter)
            tb_writer.add_scalar('disparity_smoothness_loss', loss_2.item(), n_iter)
            tb_writer.add_scalar('geometry_consistency_loss', loss_3.item(), n_iter)
            tb_writer.add_scalar('total_loss', loss.item(), n_iter)

        # record loss and EPE
        losses.update(loss.item(), args.batch_size)

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i >= epoch_size - 1:
            break

        n_iter += 1

    return losses.avg[0]

@torch.no_grad()
def validate_without_gt(args, val_loader, disp_net, pose_net, epoch, tb_writer, sample_nb_to_log=15):
    global device
    batch_time = AverageMeter()
    losses = AverageMeter(i=2, precision=4)
    log_outputs = sample_nb_to_log > 0
    # Output the logs throughout the whole dataset
    batches_to_log = list(np.linspace(0, len(val_loader), sample_nb_to_log).astype(int))
    w1, w2, w3 = args.photo_loss_weight, args.smooth_loss_weight, args.geometry_consistency_weight

    # switch to evaluate mode
    disp_net.eval()
    pose_net.eval()

    end = time.time()
    
    for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv) in enumerate(val_loader):
        tgt_img = tgt_img.to(device)
        ref_imgs = [img.to(device) for img in ref_imgs]
        intrinsics = intrinsics.to(device)
        intrinsics_inv = intrinsics_inv.to(device)


        poses, poses_inv = compute_pose_with_inv(pose_net, tgt_img, ref_imgs,)

        # compute output
        
        tgt_depth, ref_depths, disp = compute_depth(disp_net, tgt_img, ref_imgs, poses, poses_inv, intrinsics, args)

        if log_outputs and i in batches_to_log:  # log first output of wanted batches
            index = batches_to_log.index(i)
            if epoch == 0:
                for j, ref in enumerate(ref_imgs):
                    tb_writer.add_image('val Input {}/{}'.format(j, index), tensor2array(tgt_img[0]), 0)
                    tb_writer.add_image('val Input {}/{}'.format(j, index), tensor2array(ref[0]), 1)

            log_output_tensorboard(tb_writer, 'val', index, '', epoch, 1./disp[0], disp[0])

        loss_1, loss_3 = compute_photo_and_geometry_loss(tgt_img, ref_imgs, intrinsics, tgt_depth, ref_depths,
                                                         poses, poses_inv, args.num_scales, args.with_ssim,
                                                         args.with_mask, args.with_auto_mask, args.padding_mode)
        
        
        loss_2 = compute_smooth_loss(tgt_depth, tgt_img, ref_depths, ref_imgs)

        loss = w1*loss_1 + w2*loss_2 + w3*loss_3

     
        tb_writer.add_scalar('photometric_error', loss_1.item(), n_iter)
        tb_writer.add_scalar('disparity_smoothness_loss', loss_2.item(), n_iter)
        tb_writer.add_scalar('geometry_consistency_loss', loss_3.item(), n_iter)
        tb_writer.add_scalar('total_loss', loss.item(), n_iter)

        loss = w1*loss_1 + w2*loss_2 + w3*loss_3
       
        losses.update([loss, loss_1,])
       
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return losses.avg, ['Total loss', 'Photo loss']

@torch.no_grad()
def validate_with_gt(args, val_loader, disp_net, pose_net, epoch,  tb_writer,num_log_imgs=5):
    global device
    batch_time = AverageMeter()
    error_names = ['abs_diff', 'abs_rel', 'sq_rel', 'a1', 'a2', 'a3']
    errors = AverageMeter(i=len(error_names))
    log_outputs = True
    batches_to_log = list(np.linspace(0, len(val_loader)-1, num_log_imgs).astype(int))

    # switch to evaluate mode
    disp_net.eval()

    end = time.time()
    for i, (tgt_img, ref_imgs, intrinsics, depth) in enumerate(val_loader):

        tgt_img = tgt_img.to(device)
        depth = depth.to(device)
        ref_imgs = [img.to(device) for img in ref_imgs]
        intrinsics =intrinsics.to(device)
        
        # check gt
        if depth.nelement() == 0:
            continue

        poses, poses_inv = compute_pose_with_inv(pose_net, tgt_img, ref_imgs,)

        # compute output
        if args.dataset=="stillbox" or args.dataset=="crane":  #the validation data of the industry dataset does not contain rotation, therefore the unrotation
            k=random.randint(0,len(ref_imgs)-1)                #step can be omitted here
            output_disp = disp_net(torch.cat((tgt_img,ref_imgs[k]),1))
            output_depth= [1/disp for disp in output_disp]
        else:
            output_depth, _, output_disp = compute_depth(disp_net, tgt_img, ref_imgs, poses, poses_inv, intrinsics, args)
        output_disp=output_disp[0][:,0]
        output_depth=output_depth[0][:,0]

        if log_outputs and i in batches_to_log:
            index = batches_to_log.index(i)
            if epoch == 0:
                tb_writer.add_image('val Input/{}'.format(index), tensor2array(tgt_img[0]), 0)
                depth_to_show = depth[0]
                tb_writer.add_image('val target Depth/{}'.format(index),
                                            tensor2array(depth_to_show, max_value=10),
                                            epoch)
                depth_to_show[depth_to_show == 0] = 1000
                disp_to_show = (1/depth_to_show).clamp(0, 10)
                tb_writer.add_image('val target Disparity Normalized/{}'.format(index),
                                            tensor2array(disp_to_show, max_value=None, colormap='magma'),
                                            epoch)

            tb_writer.add_image('val Dispnet Output Normalized/{}'.format(index),
                                        tensor2array(output_disp[0], max_value=None, colormap='magma'),
                                        epoch)
            tb_writer.add_image('val Depth Output/{}'.format(index),
                                        tensor2array(output_depth[0], max_value=10),
                                        epoch)

        if depth.nelement() != output_depth.nelement():
            b, h, w = depth.size()
            output_depth = torch.nn.functional.interpolate(output_depth.unsqueeze(1), [h, w]).squeeze(1)

        errors.update(compute_errors(depth, output_depth, args.dataset))

        batch_time.update(time.time() - end)
        end = time.time()
       
    return errors.avg, error_names


def compute_depth(disp_net, tgt_img, ref_imgs, poses, poses_inv, intrinsics, args):
    '''
    Main modifications of our proposed pipeline are within this method. 
    It takes the rotation prediction and unrotates one of them to form pairs. 
    These pairs are feed into the depthnet, which estimates the disparity, which is converted to depth by using the inverse.
    '''
    if args.dataset!="stillbox":  #stillbox does not contain rotation, therefore no need to apply unrotation
        pose_matrices=[]
        pose_matrices_inv=[]
        for pose in poses:
            pose_matrices.append(pose_vec2mat(pose))  #convert pose from pitch yaw roll format to rotation matrix
        for pose in poses_inv:
            pose_matrices_inv.append(pose_vec2mat(pose))

        k=random.randint(0,len(ref_imgs)-1)  #choose random source image to pair with the target image
        ref = ref_imgs[k]
        pose = pose_matrices_inv[k]
        inv_pose= pose[:,:,:-1]
        ref_compensated = inverse_rotate(ref, inv_pose, intrinsics)    #align the rotational angle
        
        tgt_disp = disp_net(torch.cat((tgt_img,ref_compensated),1))
        tgt_depth = [1/disp for disp in tgt_disp]

        ref_depths=[]
        for i in range(len(ref_imgs)):  #predict a depth map for each source image (ref_img here)
            ref=ref_imgs[i]
            rot=pose_matrices[i][:,:,:-1]
            tgt_compensated = inverse_rotate(tgt_img, rot, intrinsics)
            ref_depth = [1/disp for disp in disp_net(torch.cat((ref,tgt_compensated),1))]
            ref_depths.append(ref_depth)
    else:
        ref_depths=[]
        tgt_disp = disp_net(torch.cat((tgt_img,ref_imgs[0]),1))
        tgt_depth = [1/disp for disp in tgt_disp]
        for ref in ref_imgs:
            ref_depth = [1/disp for disp in disp_net(torch.cat((ref,tgt_img),1))]
            ref_depths.append(ref_depth)

    return tgt_depth, ref_depths, tgt_disp

def compute_pose_with_inv(pose_net, tgt_img, ref_imgs):
    poses = []
    poses_inv = []
    for ref_img in ref_imgs:
        poses.append(pose_net(tgt_img, ref_img))
        poses_inv.append(pose_net(ref_img, tgt_img))

    return poses, poses_inv


if __name__ == '__main__':
    main()
