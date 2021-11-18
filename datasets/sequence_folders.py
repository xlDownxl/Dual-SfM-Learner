import torch.utils.data as data
import numpy as np
from imageio import imread
from path import Path
import random
import os
import cv2
import torch
def resize_intrinsics(intrinsics, target_height, target_width, img_height, img_width):
    downscale_height = target_height/img_height
    downscale_width = target_width/img_width

    intrinsics_scaled = np.concatenate((intrinsics[0]*downscale_width,intrinsics[1]*downscale_height, intrinsics[2]), axis=0).reshape(3,3)
    return intrinsics_scaled

def load_as_float(path, height, width, isnyu:False):
    img=cv2.imread(path)
    if isnyu:
        k1_rgb =  2.0796615318809061e-01;
        k2_rgb = -5.8613825163911781e-01;
        p1_rgb = 7.2231363135888329e-04;
        p2_rgb = 1.0479627195765181e-03;
        k3_rgb = 4.9856986684705107e-01;
        fx = 5.1885790117450188e+02
        fy = 5.1946961112127485e+02
        cx = 3.2558244941119034e+02 
        cy = 2.5373616633400465e+02

        intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).astype(np.float32).reshape((3, 3))

        img=cv2.undistort(img,intrinsics,np.array([k1_rgb,k2_rgb,p1_rgb,p2_rgb,k3_rgb]),None,None)
        
        img=img[16:-16,16:-16]
    
    if height!=None or width !=None:
        img=cv2.resize(img, (width,height), interpolation = cv2.INTER_AREA)  
    return img.astype(np.float32)


class SequenceFolder(data.Dataset):

    def __init__(self, root, seed=None, train=True, sequence_length=3, transform=None, skip_frames=1, dataset='kitti', width=None, height=None, downsample=0, with_gt=False):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        self.with_gt=with_gt
        self.width=width
        self.height=height
        self.train=train
        scene_list_path = self.root/'train.txt' if train else self.root/'val.txt'
        if False:#not self.train:
            self.scenes = [self.root/"lidar_data_synced/"+folder[:-1]+"/RS_color" for folder in open(scene_list_path)]
        else:
            self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        self.transform = transform
        self.dataset = dataset
        self.k = skip_frames
        self.crawl_folders(sequence_length, downsample)

    def crawl_folders(self, sequence_length,downsample):
        sequence_set = []
        demi_length = (sequence_length-1)//2
        shifts = list(range(-demi_length * self.k, demi_length * self.k + 1, self.k))
        shifts.pop(demi_length)
        for scene in self.scenes:      
            if self.dataset=="nyu-depth":
                if self.with_gt:
                    imgs = sorted(scene.files('*.jpg'))
                    depths = sorted((scene/'depth').files('*.png'))
                    intrinsics = np.genfromtxt(scene/'cam.txt').astype(np.float32).reshape((3, 3))
                else:
                    imgs = sorted(scene.files('*.ppm'))
                    imgs=imgs[::10]
                    fx = 5.1885790117450188e+02
                    fy = 5.1946961112127485e+02
                    cx = 3.2558244941119034e+02 -16
                    cy = 2.5373616633400465e+02 -16

                    intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).astype(np.float32).reshape((3, 3))

            elif self.dataset=="nyu-rectified":
                if self.with_gt:
                    imgs = sorted(scene.files('*.jpg'))
                    depths = sorted((scene/'depth').files('*.png'))
                    intrinsics = np.genfromtxt(scene/'cam.txt').astype(np.float32).reshape((3, 3))
                  
            elif self.dataset=='kitti':
                if self.with_gt:
                    imgs = sorted(scene.files('*.jpg'))
                    intrinsics = np.genfromtxt(scene/'cam.txt').astype(np.float32).reshape((3, 3))
                    depths = sorted(scene.files('*.npy'))
                else:
                    imgs = sorted(scene.files('*.jpg'))
                    intrinsics = np.genfromtxt(scene/'cam.txt').astype(np.float32).reshape((3, 3))

            elif self.dataset=='crane':
                if self.with_gt:
                    imgs = sorted(scene.files('*.png'), key = lambda x : int(str(x).split(".")[0].split("/")[-1]))
                    if len(imgs)==0:
                        imgs = sorted(scene.files('*.jpg'), key = lambda x : int(str(x).split(".")[0].split("/")[-1]))
                   
                    try:
                        intrinsics = np.genfromtxt(scene/'cam.txt').astype(np.float32).reshape((3, 3))
                    except:
                        intrinsics = np.genfromtxt(scene/'intrinsics.txt').astype(np.float32).reshape((3, 3))
                    depths = sorted(scene.files('*.npy'))
                else:
                    imgs = sorted(scene.files('*.jpg'), key = lambda x : int(str(x).split(".")[0].split("/")[-1]))
                    if len(imgs)==0:
                        imgs = sorted(scene.files('*.png'), key = lambda x : int(str(x).split(".")[0].split("/")[-1]))
                    
                    try:
                        intrinsics = np.genfromtxt(scene/'cam.txt').astype(np.float32).reshape((3, 3))
                    except:
                        intrinsics = np.genfromtxt(scene/'intrinsics.txt').astype(np.float32).reshape((3, 3))

            elif self.dataset=='tum':
                if self.with_gt:
                    imgs = sorted(scene.files('*.png'))
                    depths = sorted(Path(scene+"/../depth/").files('*.png'))
                else:
                    imgs = sorted(scene.files('*.png'))
                fx = 525.0  # focal length x
                fy = 525.0  # focal length y
                cx = 319.5  # optical center x
                cy = 239.5  # optical center y
                intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).astype(np.float32).reshape((3, 3))
                if downsample!=0:
                    imgs=imgs[::downsample]
            if (self.height!=None or self.width !=None) and len(imgs)>0:
                dummy_img =cv2.imread(imgs[0])
                if self.dataset=="nyu-depth":
                    a,b=dummy_img.shape[0:2]
                    a-=32
                    b-=32
                else:
                    a,b=dummy_img.shape[0:2]
                #print(intrinsics)
                intrinsics = resize_intrinsics(intrinsics,self.height,self.width,a,b)
                #print(intrinsics)

            if len(imgs) < sequence_length:
                continue
            for i in range(demi_length * self.k, len(imgs)-demi_length * self.k):
                sample = {'intrinsics': intrinsics, 'tgt': imgs[i], 'ref_imgs': []}
                if self.with_gt:
                    sample["depth"]=depths[i]
                for j in shifts:
                    sample['ref_imgs'].append(imgs[i+j])
                sequence_set.append(sample)
        #if self.train:
        #    random.shuffle(sequence_set)
        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_img = load_as_float(sample['tgt'], self.height, self.width,self.dataset=="nyu-depth")
        ref_imgs = [load_as_float(ref_img, self.height, self.width,self.dataset=="nyu-depth") for ref_img in sample['ref_imgs']]

        if self.with_gt:
            if self.dataset=='nyu':
                depth = torch.from_numpy(imread(sample["depth"]).astype(np.float32)).float()/5000
            elif self.dataset=='kitti':
                depth = torch.from_numpy(np.load(sample["depth"]).astype(np.float32))
            elif self.dataset=='nyu-depth':
                depth = torch.from_numpy(imread(sample["depth"]).astype(np.float32))/5000
            elif self.dataset=='nyu-rectified':
                depth = torch.from_numpy(imread(sample["depth"]).astype(np.float32))/5000
            elif self.dataset=='tum':
                depth = torch.from_numpy(imread(sample["depth"]).astype(np.float32)).float()/5000
            elif self.dataset=='crane':
                depth = torch.from_numpy(np.load(sample["depth"]).astype(np.float32))
        else:
            depth=0

        if self.transform is not None:
            imgs, intrinsics = self.transform([tgt_img] + ref_imgs, np.copy(sample['intrinsics']))
            tgt_img = imgs[0]
            ref_imgs = imgs[1:]
        else:
            intrinsics = np.copy(sample['intrinsics'])      
        
        return tgt_img, ref_imgs, intrinsics, depth

    def __len__(self):
        return len(self.samples)