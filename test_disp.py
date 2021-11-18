import torch
from skimage.transform import resize
from scipy.ndimage.interpolation import zoom
import numpy as np
from path import Path
import argparse
from tqdm import tqdm
from inverse_warp import inverse_rotate, pose_vec2mat
from models import DispResNet, PoseResNet
import matplotlib as mpl
import matplotlib.cm as cm
import cv2
import os

parser = argparse.ArgumentParser(description='Script for DispNet testing with corresponding groundTruth',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--pretrained-dispnet", required=True, type=str, help="pretrained DispNet path")
parser.add_argument("--pretrained-posenet", default=None, type=str, help="pretrained PoseNet path (for scale factor)")
parser.add_argument("--img-height", default=480, type=int, help="Image height")
parser.add_argument("--img-width", default=640, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")
parser.add_argument("--min-depth", default=1e-3)
parser.add_argument("--max-depth", default=80, type=int)
parser.add_argument("--step", default=1, type=int)

parser.add_argument("--dataset-dir", default='.', type=str, help="Dataset directory")
parser.add_argument("--dataset-list", default=None, type=str, help="Dataset list file")
parser.add_argument("--output-dir", default=None, type=str, help="Output directory for saving predictions in a big 3D numpy file")
parser.add_argument("--depth-list", help="images extensions to glob")

parser.add_argument("--gps", '-g', action='store_true',
                    help='if selected, will get displacement from GPS for KITTI. Otherwise, will integrate speed')
parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")
parser.add_argument("--dataset", default='KITTI', help="images extensions to glob")
parser.add_argument("--downsample", default='1',type=int, help="")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
import numpy as np
from scipy import ndimage as nd

def fill(data, invalid=None):
    """
    Replace the value of invalid 'data' cells (indicated by 'invalid') 
    by the value of the nearest valid data cell

    Input:
        data:    numpy array of any dimension
        invalid: a binary array of same shape as 'data'. 
                 data value are replaced where invalid is True
                 If None (default), use: invalid  = np.isnan(data)

    Output: 
        Return a filled array. 
    """    
    if invalid is None: invalid = np.isnan(data)

    ind = nd.distance_transform_edt(invalid, 
                                    return_distances=False, 
                                    return_indices=True)
    return data[tuple(ind)]

@torch.no_grad()
def main():
    args = parser.parse_args()
    if args.dataset == 'KITTI':
        from kitti_eval.depth_evaluation_utils import test_framework_KITTI as test_framework
    elif args.dataset == 'stillbox':
        from stillbox_eval.depth_evaluation_utils import test_framework_stillbox as test_framework
    elif args.dataset == 'nyu':
        from nyu_eval.depth_evaluation_utils import test_framework_nyu as test_framework
    elif args.dataset == 'crane':
        from crane_eval.depth_evaluation_utils import test_framework_crane as test_framework

    disp_net = DispResNet(num_input_images=2).to(device)
    weights = torch.load(args.pretrained_dispnet)
    disp_net.load_state_dict(weights['state_dict'])
    disp_net.eval()
    if args.output_dir is not None:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    print("pretrained dispnet: "+args.pretrained_dispnet)
    
    if args.pretrained_posenet is None:
        print('no PoseNet specified, scale_factor will be determined by median ratio, which is kiiinda cheating\
            (but consistent with original paper)')
        seq_length = 3
    else:
        print("pretrained posenet: "+args.pretrained_posenet)
        weights = torch.load(args.pretrained_posenet)
        seq_length = 3
        pose_net = PoseResNet().to(device)
        pose_net.load_state_dict(weights['state_dict'], strict=False)

    dataset_dir = Path(args.dataset_dir)
    if args.dataset_list is not None:
        with open(args.dataset_list, 'r') as f:
            test_files = list(f.read().splitlines())
        
    else:
        test_files = [file.relpathto(dataset_dir) for file in sum([dataset_dir.files('*.{}'.format(ext)) for ext in args.img_exts], [])]

    framework = test_framework(dataset_dir, test_files, seq_length,
                               args.min_depth, args.max_depth,
                               use_gps=args.gps, step=args.step, downsample=args.downsample)

    print('{} files to test'.format(len(framework)))
    errors = np.zeros((2, 9, len(framework)), np.float32)
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.makedirs_p()

    if args.dataset=="KITTI":   #resize intrinsics of default image size is changed
        intrinsics = np.genfromtxt('kitti_eval/intrinsics.txt').astype(np.float32).reshape((3, 3))
        intrinsics=torch.from_numpy(resize_intrinsics(intrinsics,args.img_height,args.img_width,128,416)).to(device).unsqueeze(0)
    elif args.dataset=="nyu":
        intrinsics = np.genfromtxt('nyu_eval/intrinsics.txt').astype(np.float32).reshape((3, 3))
        intrinsics=torch.from_numpy(resize_intrinsics(intrinsics,args.img_height,args.img_width,460,640)).to(device).unsqueeze(0)
    elif args.dataset=="crane":
        intrinsics = np.genfromtxt('crane_eval/cam.txt').astype(np.float32).reshape((3, 3))
        intrinsics=torch.from_numpy(resize_intrinsics(intrinsics,args.img_height,args.img_width,360,640)).to(device).unsqueeze(0)


    for j, sample in enumerate(framework):
        tgt_img = sample['tgt']
        ref_imgs = sample['ref']
        
        if args.dataset=="stillbox":
            intrinsics=torch.from_numpy(sample["intrinsics"]).unsqueeze(0).to(device)

        h,w,_ = tgt_img.shape
        if (not args.no_resize) and (h != args.img_height or w != args.img_width):
            tgt_img = resize(tgt_img, (args.img_height, args.img_width)).astype(np.float32)
            ref_imgs = [resize(img, (args.img_height, args.img_width)).astype(np.float32) for img in ref_imgs]

        tgt_img = np.transpose(tgt_img, (2, 0, 1))
        ref_imgs = [np.transpose(img, (2,0,1)) for img in ref_imgs]

        tgt_img = torch.from_numpy(tgt_img).unsqueeze(0)
        tgt_img = ((tgt_img/255 - 0.45)/0.225).to(device)

        for i, img in enumerate(ref_imgs):
            img = torch.from_numpy(img).unsqueeze(0)
            img = ((img/255 - 0.45)/0.225).to(device)
            ref_imgs[i] = img

        if args.dataset=="crane":
            pred_disps = compute_depth_crane(disp_net, tgt_img, ref_imgs)   #industry evaluation data contains no rotation, th eunrotation step can be omitted
        else:
            pred_disps = compute_depth(disp_net, tgt_img, ref_imgs, pose_net, intrinsics)

        pred_disps=[pred_disp[0][0,0].cpu().numpy() for pred_disp in pred_disps]

        gt_depth = sample['gt_depth']  
        
        pred_depths = [1/pred_disp for pred_disp in pred_disps]
        pred_depths_zoomed = [zoom(pred_depth,
                                 (gt_depth.shape[0]/pred_depth.shape[0],
                                  gt_depth.shape[1]/pred_depth.shape[1])
                                 )  for pred_depth in pred_depths]

        if sample['mask'] is not None:
            mask=sample['mask']
            pred_depths_masked = [pred_depth_zoomed[mask] for pred_depth_zoomed in pred_depths_zoomed]
            gt_depth = gt_depth[mask]   

        scale_factors = [np.median(gt_depth)/np.median(pred_depth_masked) for pred_depth_masked in pred_depths_masked]        
        all_errors=np.array([compute_errors(gt_depth, (pred_depth_masked*scale_factor).clip(args.min_depth, args.max_depth)) for (pred_depth_masked,scale_factor) in zip(pred_depths_masked,scale_factors)])
        
        abs_rel=np.min(all_errors[:,1]) 
        index=np.argmax(all_errors[:,1]==abs_rel)
        errors[1,:,j] = all_errors[index]

        cat_img = 0
        if  args.output_dir is not None:
            pred_depth_zoomed=pred_depths_zoomed[index]
            scale_factor=scale_factors[index]
            cat_img = np.zeros((h, 3*w, 3))
            cat_img[:, :w] = sample['tgt']
            pred = (pred_depth_zoomed*scale_factor).clip(args.min_depth, args.max_depth)

            gt = sample['gt_depth']
            invalid=gt==0
            gt=fill(gt,invalid)

            vis_pred, vis_gt = depth_pair_visualizer(pred, gt)
            cat_img[:, w:2*w] = vis_pred
            cat_img[:, 2*w:3*w] = vis_gt
            cat_img = cat_img.astype(np.uint8)
            png_path = os.path.join(args.output_dir, "{:04}.png".format(j))
            cv2.imwrite(png_path, cv2.cvtColor(cat_img, cv2.COLOR_RGB2BGR))

    mean_errors = errors.mean(2)
    error_names = ['abs_diff', 'abs_rel','sq_rel','rms','log_rms', 'abs_log', 'a1','a2','a3']
    if args.pretrained_posenet and args.dataset=="KITTI":
        print("Results with scale factor determined by PoseNet : ")
        print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format(*error_names))
        print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(*mean_errors[0]))

    print("Results with scale factor determined by GT/prediction ratio (like the original paper) : ")
    print("{:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}".format(*error_names))
    print("{:10.4f} {:10.4f} {:10.4f} {:10.4f} {:10.4f} {:10.4f} {:10.4f} {:10.4f} {:10.4f}".format(*mean_errors[1]))

   
def compute_depth(disp_net, tgt_img, ref_imgs, pose_net, intrinsics):
    disps=[]
    for ref_img in ref_imgs:
        pose = pose_net(ref_img, tgt_img) 
        pose=pose_vec2mat(pose) 

        pose= pose[:,:,:-1]
        ref_compensated = inverse_rotate(ref_img, pose, intrinsics)

        tgt_disp = disp_net(torch.cat((tgt_img,ref_compensated),1))
        disps.append(tgt_disp)
    return  disps

def resize_intrinsics(intrinsics, target_height, target_width, img_height, img_width):
    downscale_height = target_height/img_height
    downscale_width = target_width/img_width

    intrinsics_scaled = np.concatenate((intrinsics[0]*downscale_width,intrinsics[1]*downscale_height, intrinsics[2]), axis=0).reshape(3,3)
    return intrinsics_scaled

def depth_pair_visualizer(pred, gt):
    inv_pred = 1 / (pred + 1e-6)
    inv_gt = 1 / (gt + 1e-6)

    vmax = np.percentile(inv_gt, 95)
    normalizer = mpl.colors.Normalize(vmin=inv_gt.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')

    vis_pred = (mapper.to_rgba(inv_pred)[:, :, :3] * 255).astype(np.uint8)
    vis_gt = (mapper.to_rgba(inv_gt)[:, :, :3] * 255).astype(np.uint8)

    return vis_pred, vis_gt

def compute_pose_with_inv(pose_net, tgt_img, ref_imgs):
    poses = []
    for ref_img in ref_imgs:
        poses.append(pose_net(tgt_img, ref_img))

    return poses

def compute_depth_crane(disp_net, tgt_img, ref_imgs, ):
    disps=[]
    for ref_img in ref_imgs:
  
        tgt_disp = disp_net(torch.cat((tgt_img,ref_img),1))
        disps.append(tgt_disp)   
    return disps

def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_log = np.mean(np.abs(np.log(gt) - np.log(pred)))

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    abs_diff = np.mean(np.abs(gt - pred))

    sq_rel = np.mean(((gt - pred)**2) / gt)

    return abs_diff, abs_rel, sq_rel, rmse, rmse_log, abs_log, a1, a2, a3

if __name__ == '__main__':
    main()
