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
from matplotlib import pyplot as plt

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
    error=0
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
                               use_gps=args.gps, step=args.step,downsample=4)

    print('{} files to test'.format(len(framework)))
    errors = np.zeros((2, 9, len(framework)), np.float32)
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.makedirs_p()
    def depth_pair_visualizer( gt):
        """
        Args:
            data (HxW): depth data
        Returns:
            vis_data (HxWx3): depth visualization (RGB)
        """
        

       
        inv_gt = 1 / (gt + 1e-6)
        vmax = np.percentile(inv_gt, 95)
        normalizer = mpl.colors.Normalize(vmin=inv_gt.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')

        
        vis_gt = (mapper.to_rgba(inv_gt)[:, :, :3] * 255).astype(np.uint8)

        return  vis_gt
        
    for j, sample in tqdm(enumerate(framework)):


        gt_depth = sample['gt_depth']
        #invalid=gt_depth==0
        
        #gt_depth=fill(gt_depth,invalid)
        #gt_depth = depth_pair_visualizer( gt_depth)

        #png_path = os.path.join(args.output_dir, "{:04}.png".format(j))
        #print(png_path)
        #cv2.imwrite(png_path, cv2.cvtColor(gt_depth, cv2.COLOR_RGB2BGR))
        if sample['mask'] is not None:
            mask=sample['mask']
            gt_depth = gt_depth[mask]

        pred_depth_masked=np.ones(gt_depth.shape).astype(np.float32)
        
        scale_factor = np.median(gt_depth)/np.median(pred_depth_masked) 
        #print(scale_factor)


       # png_path1 = os.path.join(args.output_dir, "{:04}.png".format(j))
       # png_path2 = os.path.join(args.output_dir, "{:04}rgb.png".format(j))
        #diff_img=np.square(np.abs(gt_depth- (pred_depth_masked*scale_factor).clip(args.min_depth, args.max_depth)))
        #error+=np.mean(diff_img)
        #print(np.mean(diff_img))
        #print(diff_img[200:210,200:210])
        #diff_img=diff_img/np.max(diff_img)*255
        
        #print(diff_img[200:210,200:210])
        
        #diff_img=diff_img/np.max(diff_img)*255
        #cv2.imwrite(png_path2, cv2.cvtColor(sample["tgt"], cv2.COLOR_RGB2BGR))
        #plt.imshow(gt_depth)
        #plt.colorbar()
        #plt.savefig(png_path1)
        #plt.close()
        #cv2.imwrite(png_path1, diff_img)

        all_errors=compute_errors(gt_depth, (pred_depth_masked*scale_factor).clip(args.min_depth, args.max_depth))
       
        errors[1,:,j] = all_errors
    print("error")
    print(error/len(framework))
    mean_errors = errors.mean(2)
    error_names = ['abs_diff', 'abs_rel','sq_rel','rms','log_rms', 'abs_log', 'a1','a2','a3']
    if args.pretrained_posenet and args.dataset=="KITTI":
        print("Results with scale factor determined by PoseNet : ")
        print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format(*error_names))
        print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(*mean_errors[0]))

    print("Results with scale factor determined by GT/prediction ratio (like the original paper) : ")
    print("{:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}".format(*error_names))
    print("{:10.4f} {:10.4f} {:10.4f} {:10.4f} {:10.4f} {:10.4f} {:10.4f} {:10.4f} {:10.4f}".format(*mean_errors[1]))

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
