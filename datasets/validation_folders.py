import torch.utils.data as data
import numpy as np
from imageio import imread
from path import Path
import os
import torch
import cv2

def load_as_float(path, height, width):
    img=cv2.imread(path)
    if height!=None or width !=None:
        img=cv2.resize(img, (width,height), interpolation = cv2.INTER_AREA)  
    return img.astype(np.float32)

def crawl_folders(folders_list, dataset, k, downsample, sequence_length):
        imgs = []
        depths = []
        sequence_set = []
        for folder in folders_list:
            if dataset == 'nyu-rectified':
                current_depth = sorted((folder/'depth/').files('*.png'))
                current_imgs = sorted(folder.files('*.jpg'))
            elif dataset == 'kitti' or dataset=="crane":
                current_imgs = sorted(folder.files('*.png'))
                current_depth = sorted(folder.files('*.npy'))
            elif dataset == 'nyu-depth':
                current_imgs = sorted(folder.files('*.jpg'))
                current_depth = sorted((folder/'depth').files('*.png'))
            elif dataset == 'tum':
                current_imgs = sorted(folder.files('*.png'))
                current_depth = sorted(Path(folder+"/../depth/").files('*.png'))
            imgs.extend(current_imgs)
            depths.extend(current_depth)
            intrinsics = np.genfromtxt(folder/'cam.txt').astype(np.float32).reshape((3, 3))
        #if downsample!=0:
        #    imgs=imgs[::downsample]
        #    depths=depths[::downsample]
        demi_length = (sequence_length-1)//2
        shifts = list(range(-demi_length * k, demi_length * k + 1, k))
        shifts.pop(demi_length) 
        for i in range(demi_length * k, len(imgs)-demi_length * k):
                sample = {'intrinsics': intrinsics, 'tgt': imgs[i],'depth': depths[i], 'ref_imgs': []}
                for j in shifts:
                    sample['ref_imgs'].append(imgs[i+j])
                sequence_set.append(sample)
        #for i in range(k,len(imgs)):
        #    sample = {'tgt': imgs[i],'depth': depths[i], 'ref': imgs[i-k]}
        #    sequence_set.append(sample)

        return sequence_set


class ValidationSet(data.Dataset):
    def __init__(self, root, transform=None, dataset='kitti', skip_frames=1, downsample=0, sequence_length=3):
        self.root = Path(root)
        scene_list_path = self.root/'val.txt'
        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        self.transform = transform
        self.dataset = dataset
        self.samples= crawl_folders(self.scenes, self.dataset, skip_frames, downsample, sequence_length)

    def __getitem__(self, index):
        sample=self.samples[index]
        tgt_img = imread(sample["tgt"]).astype(np.float32)
        #ref = imread(sample["ref"]).astype(np.float32)
        ref_imgs = [load_as_float(ref_img, None, None) for ref_img in sample['ref_imgs']]

        if self.dataset=='nyu':
            depth = torch.from_numpy(imread(sample["depth"]).astype(np.float32)).float()/5000
        elif self.dataset=='kitti':
            depth = torch.from_numpy(np.load(sample["depth"]).astype(np.float32))
        elif self.dataset=='nyu-depth':
            depth = torch.from_numpy(imread(sample["depth"]).astype(np.float32))/5000
        elif self.dataset=='nyu-rectified':
            depth = torch.from_numpy(imread(sample["depth"]).astype(np.float32))
            #img=img[10:-10,10:-10]
            #ref=ref[10:-10,10:-10]
        elif self.dataset=='tum':
            depth = torch.from_numpy(imread(sample["depth"]).astype(np.float32)).float()/5000

        if self.transform is not None:
            imgs, intrinsics = self.transform([tgt_img] + ref_imgs, np.copy(sample['intrinsics']))
            tgt_img = imgs[0]
            ref_imgs = imgs[1:]
        #try:
        #    torch.median(depth)
        #except:
        #    print(sample["tgt"])
  
        return tgt_img, ref_imgs, intrinsics, depth

    def __len__(self):
        return len(self.samples)
