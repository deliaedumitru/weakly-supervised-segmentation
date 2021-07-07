import os
from skimage import io
import pandas as pd
from utils import segmap_rgb_to_classes
import numpy as np
import torch
from torch.utils.data import Dataset

class VaihingenDataset(Dataset):
    
    def __init__(self, img_dir, gt_dir, labels_f, strong_idx, weak_idx=[], transform=None):

        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.labels_f = labels_f
        
        self.strong_idx = strong_idx
        self.weak_idx = weak_idx
        
        self.labels = pd.read_csv(labels_f, header=None)

        self.transform = transform

    def __len__(self):
        return len(self.strong_idx) + len(self.weak_idx)

    def __getitem__(self, idx):
        
        f_name = self.labels.iloc[idx][0]

        img_name = os.path.join(self.img_dir, '{}.png'.format(f_name))
        gt_name = os.path.join(self.gt_dir, '{}.png'.format(f_name))
            
        labels = np.array(self.labels.iloc[idx][1:])
        
        img = io.imread(img_name) 
        gt = segmap_rgb_to_classes(io.imread(gt_name))
        
        img_float = img / 255.
        gt_float = gt
        
        if idx in self.weak_idx:
            weak_sup = True
        else:
            weak_sup = False
        
        sample = {
                'image': img_float.astype(np.float32),
                'ground_truth': gt_float.astype(np.float32),
                'labels': labels.astype(np.float32),
                'weak_supervision': np.array(weak_sup)
            }
                
        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    
    def __call__(self, sample):
        img, gt, labels, weak_sup = sample['image'], sample['ground_truth'], sample['labels'], sample['weak_supervision']

        # swapping axes for images
        img = img.transpose((2, 0, 1))
        gt = gt.transpose((2, 0, 1))
        
        return {
            'image': torch.from_numpy(img),
            'ground_truth': torch.from_numpy(gt),
            'labels': torch.from_numpy(labels),
            'weak_supervision': torch.from_numpy(weak_sup)
        }