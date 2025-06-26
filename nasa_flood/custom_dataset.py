import os
import tifffile
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_dir = os.path.join(root_dir, 'images')
        self.label_dir = os.path.join(root_dir, 'labels')
        self.image_filenames = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.tif')])

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
        image = Image.fromarray(image.astype('uint8'), 'RGB')

        label_mask = Image.open(label_path)
        
        # Convert mask to binary label: 1 if flood (any non-black pixel), 0 otherwise
        label = 1 if np.any(np.array(label_mask) > 0) else 0

        if self.transform:
            image = self.transform(image)
            
        return image, label 