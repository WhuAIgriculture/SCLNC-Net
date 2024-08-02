import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps

class BasicDataset(Dataset):
    def __init__(self, imgs_dir, patch_size=128, patch_num=1):
        self.imgs_dir = imgs_dir
        self.patch_size = patch_size
        self.patch_num = patch_num
        
        self.img_list = [os.path.join(imgs_dir, file) for file in os.listdir(imgs_dir) if not file.startswith('.')]
    
    def __len__(self):
        return len(self.img_list)
    
    @classmethod
    def preprocess(cls, pil_img, patch_size, patch_coords, flip_op):
        if flip_op == 1:
            pil_img = ImageOps.mirror(pil_img)
        elif flip_op == 2:
            pil_img = ImageOps.flip(pil_img)
        
        img_nd = np.array(pil_img)
        assert len(img_nd.shape) == 3, 'Training/validation images should be 3 channels colored images'
        img_nd = img_nd[patch_coords[1]:patch_coords[1]+patch_size, patch_coords[0]:patch_coords[0]+patch_size, :]
        
        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        img_trans = img_trans / 255
        
        return img_trans
    
    
    def __getitem__(self, i):
        img_file = self.img_list[i]
        
        image_input = Image.open(img_file)
        # get image size
        w, h = image_input.size
        
        suffix_img = img_file.split('.')[-1]
        suffix_gt = 'gt.' + suffix_img
        
        # get ground truth images
        parts = img_file.split('_')
        prefix_img = ''
        for i in range(len(parts) - 1):
            prefix_img += parts[i] + '_'
        
        gt_file = prefix_img + suffix_gt
        image_gt = Image.open(gt_file)
        
        # get flipping option
        flip_op = np.random.randint(3)
        # get random patch coord
        patch_x = np.random.randint(0, w - self.patch_size + 1)
        patch_y = np.random.randint(0, h - self.patch_size + 1)
        image_input_patch = self.preprocess(image_input, self.patch_size, (patch_x, patch_y), flip_op)
        image_gt_patch = self.preprocess(image_gt, self.patch_size, (patch_x, patch_y), flip_op)
        
        for j in range(self.patch_num - 1):
            # get flipping option
            flip_op = np.random.randint(3)
            # get random patch coord
            patch_x = np.random.randint(0, w - self.patch_size + 1)
            patch_y = np.random.randint(0, h - self.patch_size + 1)
            
            temp_input = self.preprocess(image_input, self.patch_size, (patch_x, patch_y), flip_op)
            temp_gt = self.preprocess(image_gt, self.patch_size, (patch_x, patch_y), flip_op)
            image_input_patch = np.append(image_input_patch, temp_input, axis=0)
            image_gt_patch = np.append(image_gt_patch, temp_gt, axis=0)
        
        return {'image_input': torch.from_numpy(image_input_patch), 
                'image_gt': torch.from_numpy(image_gt_patch)}