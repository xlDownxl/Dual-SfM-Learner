import torch
import cv2
from cv2 import resize
import numpy as np
from path import Path
import argparse
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.cm as cm
import cv2
from models import DispResNet
from utils import tensor2array
import os
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

parser = argparse.ArgumentParser(description='Inference script for DispNet learned with \
                                 Structure from Motion Learner inference on KITTI Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--output-disp", action='store_true', help="save disparity img")
parser.add_argument("--output-depth", action='store_true', help="save depth img")
parser.add_argument("--pretrained", required=True, type=str, help="pretrained DispResNet path")
parser.add_argument("--img-height", default=256, type=int, help="Image height")
parser.add_argument("--img-width", default=320, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")

parser.add_argument("--dataset-list", default=None, type=str, help="Dataset list file")
parser.add_argument("--dataset-dir", default='.', type=str, help="Dataset directory")
parser.add_argument("--output-dir", default='output', type=str, help="Output directory")
parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")
parser.add_argument('--resnet-layers', type=int, default=18, choices=[18, 50],
                    help='depth network architecture.')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def high_res_colormap(low_res_cmap, resolution=1000, max_value=1):
    # Construct the list colormap, with interpolated values for higer resolution
    # For a linear segmented colormap, you can just specify the number of point in
    # cm.get_cmap(name, lutsize) with the parameter lutsize
    x = np.linspace(0, 1, low_res_cmap.N)
    low_res = low_res_cmap(x)
    new_x = np.linspace(0, max_value, resolution)
    high_res = np.stack([np.interp(new_x, x, low_res[:, i])
                         for i in range(low_res.shape[1])], axis=1)
    return ListedColormap(high_res)


def opencv_rainbow(resolution=1000):
    # Construct the opencv equivalent of Rainbow
    opencv_rainbow_data = (
        (0.000, (1.00, 0.00, 0.00)),
        (0.400, (1.00, 1.00, 0.00)),
        (0.600, (0.00, 1.00, 0.00)),
        (0.800, (0.00, 0.00, 1.00)),
        (1.000, (0.60, 0.00, 1.00))
    )

    return LinearSegmentedColormap.from_list('opencv_rainbow', opencv_rainbow_data, resolution)


COLORMAPS = {'rainbow': opencv_rainbow(),
             'magma': high_res_colormap(cm.get_cmap('magma')),
             'bone': cm.get_cmap('bone', 10000)}
@torch.no_grad()
def main():
    args = parser.parse_args()
    if not(args.output_disp or args.output_depth):
        print('You must at least output one value !')
        return

    disp_net = DispResNet(args.resnet_layers, False, num_input_images=1).to(device)
    weights = torch.load(args.pretrained)
    disp_net.load_state_dict(weights['state_dict'])
    disp_net.eval()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.makedirs_p()

    if args.dataset_list is not None:
        with open(args.dataset_list, 'r') as f:
            test_files = [dataset_dir/file for file in f.read().splitlines()]
    else:
        test_files = sorted(sum([dataset_dir.files('*.{}'.format(ext)) for ext in args.img_exts], []))

    print('{} files to test'.format(len(test_files)))

    for i in tqdm(range(1,len(test_files))):
        file=test_files[i]
        img = cv2.imread(file).astype(np.float32)
        #print(img.shape)
       
        #ref_img = cv2.imread(test_files[i-1]).astype(np.float32)

        h, w, _ = img.shape
        #print(h)
        #print(w)
        if (not args.no_resize) and (h != args.img_height or w != args.img_width):
            img = resize(img, ( args.img_width,args.img_height,)).astype(np.float32)
        #print(img.shape)
        #cv2.imwrite("lel.png",img)
        img_orig=img.copy()
        img = np.transpose(img, (2, 0, 1))

        tensor_img = torch.from_numpy(img).unsqueeze(0)
        tensor_img = ((tensor_img/255 - 0.45)/0.225).to(device)

       
        output = disp_net(tensor_img)[0][0][0].cpu().numpy()
        depth=1/output

        
        #scale_factor = 3/np.median(depth)
        #depth=depth*scale_factor*10
        #print(np.max(depth))
        #print(np.min(depth))
        
        #disparity
        #cat_img = np.zeros((args.img_height, 2*args.img_width, 3))     
        #cat_img[:, :args.img_width] =img_orig
        #pred=np.stack([(output/np.max(output))*255,(output/np.max(output))*255,(output/np.max(output))*255],axis=2)
        #cat_img[:, args.img_width:2*args.img_width] = pred
        #cat_img = cat_img.astype(np.uint8)
        #png_path = os.path.join(args.output_dir, str(i)+".png")
        #cv2.imwrite(png_path, cv2.cvtColor(cat_img, cv2.COLOR_RGB2BGR))

        #depth
        cat_img = np.zeros((args.img_height, 2*(args.img_width), 3))   


        cat_img[:, :args.img_width] =img_orig

        
        depth_normalized=depth
        #depth_normalized=(depth-np.min(depth))   #255 ist WEIß!
        depth_normalized=depth_normalized/np.max(depth_normalized)*255
        #depth_normalized[depth_normalized>200]=255
        #print(depth_normalized[10,-10:])
        #print(depth_normalized[10,:10])
        #print("_________")
        #print(depth_normalized.shape)
        
        #pred=COLORMAPS['magma'](depth_normalized).astype(np.uint8)*255
        #print(pred.shape)
        pred=np.stack([depth_normalized,depth_normalized,depth_normalized],axis=2) 
    
        cat_img[:, args.img_width:] = pred
        cat_img = cat_img.astype(np.uint8)
        png_path = os.path.join(args.output_dir, str(i)+".png")     
        cv2.imwrite(png_path, cv2.cvtColor(cat_img, cv2.COLOR_RGB2BGR))

        #if args.output_disp:
        #    disp = (255*tensor2array(output, max_value=None, colormap='bone')).astype(np.uint8)
        #    imsave(output_dir/'{}_disp{}'.format(file_name, ".png"), np.transpose(disp, (1, 2, 0)))
        #if args.output_depth:
        #    depth = 1/output
        #    depth = (255*tensor2array(depth, max_value=10, colormap='rainbow')).astype(np.uint8)
        #    imsave(output_dir/'{}_depth{}'.format(file_name, ".png"), np.transpose(depth, (1, 2, 0)))

if __name__ == '__main__':
    main()
