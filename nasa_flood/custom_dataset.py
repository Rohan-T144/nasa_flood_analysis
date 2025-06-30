import os
import tifffile
import numpy as np
from PIL import Image as Image_PIL
from torch.utils.data import Dataset
import torch

from torchvision.tv_tensors import Image, Mask

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, flag=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_dir = os.path.join(root_dir, 'images')
        self.label_dir = os.path.join(root_dir, 'labels')
        self.image_filenames = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.tif')], key=lambda name : int(name[:-4])) #sort by increasing numbers
        self.flag = flag

        # if train or val flag given, use first 80% for train and last 20% for val
        if self.flag == "train":
            self.image_filenames = self.image_filenames[:int(len(self.image_filenames) * 0.8)]
        elif self.flag=="val":
            self.image_filenames = self.image_filenames[int(len(self.image_filenames) * 0.8):]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_name = self.image_filenames[idx]
        image_path = os.path.join(self.image_dir, image_name)
        
        # Construct label path
        label_name = os.path.splitext(image_name)[0] + '.png'
        label_path = os.path.join(self.label_dir, label_name)

        # Load image and label
        image = tifffile.imread(image_path)
        
        # Convert to 3 channels if it's a single channel image
        if len(image.shape) == 2:
            image = np.stack([image]*3, axis=-1)
        
        # Convert to PIL Image for transforms
        image = Image_PIL.fromarray(image.astype('uint8'), 'RGB')

        label_mask = Image_PIL.open(label_path)
        
        # Convert mask to binary label: 1 if flood (any non-black pixel), 0 otherwise
        label = 1 if np.any(np.array(label_mask) > 0) else 0

        if self.transform:
            image = self.transform(image)
            
        return image, label 

class CustomDatasetSeg(Dataset):
    def __init__(self, root_dir, transform=None, mask_transform = None, flag=None):
        self.root_dir = root_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.image_dir = os.path.join(root_dir, 'images')
        self.label_dir = os.path.join(root_dir, 'labels')
        self.image_filenames = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.tif')], key=lambda name : int(name[:-4])) #sort by increasing numbers
        self.flag = flag

        # if train or val flag given, use first 80% for train and last 20% for val
        if self.flag == "train":
            self.image_filenames = self.image_filenames[:int(len(self.image_filenames) * 0.8)]
        elif self.flag=="val":
            self.image_filenames = self.image_filenames[int(len(self.image_filenames) * 0.8):]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_name = self.image_filenames[idx]
        image_path = os.path.join(self.image_dir, image_name)
        
        # Construct label path
        label_name = os.path.splitext(image_name)[0] + '.png'
        label_path = os.path.join(self.label_dir, label_name)

        # Load image and label
        image = tifffile.imread(image_path)
        
        # Convert to 3 channels if it's a single channel image
        if len(image.shape) == 2:
            image = np.stack([image]*3, axis=-1)
        
        # Convert to PIL Image for transforms
        
        #get only the rgb channels
        rgb = image[:,:,:3] 
        #scale down values to uint8 range (was likely uint16)
        rgb_scaled = ((rgb - rgb.min()) / (rgb.max() - rgb.min()) * 255).astype(np.uint8) 

        image = Image_PIL.fromarray(rgb_scaled, 'RGB') #128, 128, 3
        label_mask = Image_PIL.open(label_path) #128, 128
        

        # image = Image(Image_PIL.fromarray(rgb_scaled, 'RGB')) #128, 128, 3
        # label_mask = Mask(Image_PIL.open(label_path)) #128, 128

        if self.transform:
            image = self.transform(image) # 3, 224, 224

        if self.mask_transform:
            label_mask = self.mask_transform(label_mask) # 1, 224, 224
        
        label_mask = torch.from_numpy(np.array(label_mask) )
            
        return image, label_mask