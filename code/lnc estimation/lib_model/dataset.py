import os
import numpy as np
import torch
from PIL import ImageOps, Image
from torch.utils.data import Dataset


class BasicDataset(Dataset):
    def __init__(self, path_dataset, patch_size=128, patch_num=1, mode='train'):
        self.path_dataset = path_dataset
        self.patch_size = patch_size
        self.patch_num = patch_num
        self.mode = mode
        
        valid_images = ('.jpg', '.jpeg', '.png')
        self.img_files = []
        for file in os.listdir(path_dataset):
          if file.lower().endswith(valid_images):
            self.img_files.append(os.path.join(path_dataset, file))
        
    def __len__(self):
        return len(self.img_files)
    
    @classmethod
    ## image preprocess (mirror, flip and crop)
    def preprocess(cls, img, patch_size, patch_pos, flag_flip):
        if flag_flip == 1:
            img = ImageOps.mirror(img)
        elif flag_flip == 2:
            img = ImageOps.flip(img)
        
        img_crop = np.array(img)
        img_crop = img_crop[patch_pos[1]: patch_pos[1] + patch_size, patch_pos[0]: patch_pos[0] + patch_size, :]
        img_trans = img_crop.transpose((2, 0, 1))
        img_trans = img_trans / 255
        return img_trans
    
    def __getitem__(self, item):
        img_file = self.img_files[item]
        all_img = Image.open(img_file)
        w, h = all_img.size
        name_image = img_file.split(self.path_dataset)[-1][1:]
        
        ## acquire label value
        lnc = float(name_image[21:24]) / 100
        
        patch_size = self.patch_size
        patch_num = self.patch_num
        if self.mode == 'val':
            flag_flip = 0
            patch_size = min([w, h])
            patch_num = 0
        else:
            flag_flip = np.random.randint(3)
        
        patch_x = np.random.randint(0, high = w - patch_size + 1)
        patch_y = np.random.randint(0, high = h - patch_size + 1)
        all_img_patches = self.preprocess(all_img, patch_size, (patch_x, patch_y), flag_flip)
        lnc_patches = np.array([lnc])
        
        ## acquire patch_num imgae
        for i in range(patch_num - 1):
            flag_flip = np.random.randint(3)
            patch_x = np.random.randint(0, high = w - patch_size + 1)
            patch_y = np.random.randint(0, high = h - patch_size + 1)
            temp = self.preprocess(all_img, patch_size, (patch_x, patch_y), flag_flip)
            all_img_patches = np.append(all_img_patches, temp, axis=0)  ## concatenate the first image and patch_num image
        
        dataset = {'image': torch.from_numpy(all_img_patches),
                   'label': torch.from_numpy(lnc_patches)}
        return dataset