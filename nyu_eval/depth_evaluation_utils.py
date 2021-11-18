import numpy as np
import json
from path import Path
from imageio import imread
from tqdm import tqdm
import os
import torch
from matplotlib import pyplot as plt

class test_framework_nyu(object):
    def __init__(self, root, test_files, seq_length=3, min_depth=1e-3, max_depth=80, step=1,use_gps=True, crop=False, downsample=1):
        self.root = root
        self.min_depth, self.max_depth = min_depth, max_depth
        self.crop=crop
        self.gt_files, self.tgt_imgs, self.ref_imgs, self.poses= read_scene_data(self.root, test_files,seq_length, step, downsample)

    def __getitem__(self, i):
        depth=imread(self.gt_files[i])/5000

        return {'tgt': imread(self.tgt_imgs[i]).astype(np.float32),
                'ref': [imread(ref).astype(np.float32) for ref in self.ref_imgs[i]],
                'gt_depth': depth,
                'poses': self.poses[i],
                'mask': generate_mask(depth, self.min_depth, self.max_depth, self.crop),             
                }
        

    def __len__(self):
        return len(self.tgt_imgs)

def read_scene_data(data_root, test_list, seq_length=5, step=3, downsample=1):
    data_root = Path(data_root)

    gt_files = []
    ref_imgs = []
    poses = []
    tgt_imgs = []

    print('getting test metadata ... ')
    for sample in test_list:
        shift_range = step * (np.arange(seq_length))
        shift_range = shift_range- shift_range[len(shift_range)//2]
        folder, file = sample.split('/')
        
        
        img_data = sorted([fn for fn in os.listdir(data_root/folder)
              if "color" in fn])
        depth_data = sorted([fn for fn in os.listdir(data_root/folder)
              if "depth" in fn])
        #folder_content=sorted(os.listdir(data_root/folder))
        scene_length = len(img_data)
        file=".".join(file.split(".")[:-1])+"_color.png"
        try:
            index=img_data.index(file)
        except:
            print("{} not found in scene {}".format(file,folder))
            continue

        tgt_img_path = data_root/(folder+"/"+file)
        folder_path = data_root/folder

        if tgt_img_path.isfile():
            # if index is high enough, take only frames before. Otherwise, take only frames after.
            if index + shift_range[0] < 0:
                middle=len(shift_range)//2
                shift_range[:middle]=shift_range[middle+1:]
                ref_indices = index + shift_range
                tgt_index = len(shift_range)//2
            elif index + shift_range[-1] >= scene_length-1:  #if tgt image is the first of sequence just take the following one as t-1 and t+1
                shift_range[middle+1:]=shift_range[:middle]
                ref_indices = index + shift_range 
                tgt_index = len(shift_range)//2
            else:
                ref_indices = index + shift_range
                tgt_index = len(shift_range)//2
            tgt_imgs.append(tgt_img_path)
            ref_img_indizes=np.delete(ref_indices,tgt_index)
            imgs_path = [folder_path/'{}'.format(img_data[ref_index]) for ref_index in ref_img_indizes]

            gt_files.append(data_root/folder+'/{}'.format(depth_data[index]))
            ref_imgs.append(imgs_path)

            
        else:
            print('{} missing'.format(tgt_img_path))
        poses.append([])



    return gt_files, tgt_imgs, ref_imgs, poses


def generate_mask(gt_depth, min_depth, max_depth, crop=False): 
    gt_height, gt_width = gt_depth.shape
    mask = np.logical_and(gt_depth > min_depth,
                          gt_depth < max_depth)
    if crop:
        #mask=mask[16:-16,16:-16]
        gt_height, gt_width = gt_depth.shape
        crop = np.array([0.09375 * gt_height, 0.985 * gt_height,
                     0.0640625 * gt_width,  0.9390625 * gt_width]).astype(np.int32)

        mask=mask[crop[0]:crop[1],crop[2]:crop[3]]
        return mask
    # crop gt to exclude border values
    # if used on gt_size 100x100 produces a crop of [-95, -5, 5, 95]
    
    crop = np.array([0.09375 * gt_height, 0.98125 * gt_height,
                     0.0640625 * gt_width,  0.9390625 * gt_width]).astype(np.int32)

    crop_mask = np.zeros(mask.shape)
    crop_mask[crop[0]:crop[1],crop[2]:crop[3]] = 1
    mask = np.logical_and(mask, crop_mask)
    return mask
