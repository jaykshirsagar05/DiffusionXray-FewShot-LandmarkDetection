


# ------------------------------------------------------------------------
#                               Libraries
# ------------------------------------------------------------------------

import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
import albumentations.pytorch

import torch
import utils
from skimage.transform import resize
import nibabel as nib  # For 3D medical imaging (NIfTI format)

# ------------------------------------------------------------------------
#                               Chest Dataset
# ------------------------------------------------------------------------

# Load the dataset from the train and test folders in the root directory
class ChestDiffusionDataset(Dataset):
    def __init__(self, root_dir, channels=1, transform=None, phase='train'):

        self.root_dir = root_dir
        self.transform = transform
        self.channels = channels
       
        self.pth_Image = os.path.join(root_dir, 'pngs')
            
        # file index
        files = [i[:-4] for i in sorted(os.listdir(self.pth_Image))]

        exclude_list = ['CHNCXR_0059_0', 'CHNCXR_0178_0', 'CHNCXR_0228_0', 'CHNCXR_0267_0', 'CHNCXR_0295_0', 'CHNCXR_0310_0', 'CHNCXR_0285_0', 'CHNCXR_0276_0', 'CHNCXR_0303_0']
        if exclude_list is not None:
            st = set(exclude_list)
            files = [f for f in files if f not in st]

        n = len(files)
        train_num = 195 
        val_num = 34  
        test_num = n - train_num - val_num
        if phase == 'train':
            self.image_files = files[:train_num+val_num]
        elif phase == 'test':
            self.image_files = files[-test_num:]
        elif phase == 'all':
            self.image_files = files
        else:
            raise Exception("Unknown phase: {phase}".format(phase=phase))
        

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_name = self.image_files[idx]

        image = self.read_image(os.path.join(self.pth_Image, image_name + '.png'))

        data_dict = {'name': image_name, 'image': image}   

        return data_dict

    def read_image(self, image_path):

        if self.channels == 3:
            image = Image.open(image_path).convert('RGB')
            image_np = np.array(image).astype(np.float32)

        elif self.channels == 1:
            image = Image.open(image_path).convert('L')
            image_np = np.array(image).astype(np.float32)
            image_np = np.expand_dims(image_np, axis=2) # add channel dimension
        else:
            raise ValueError('Channels must be either 1 or 3')
        
        if self.transform:
            image = self.transform(image=image_np)['image']
        
        return image
        
    
# ------------------------------------------------------------------------
#                               HAND Dataset
# ------------------------------------------------------------------------

# Load the dataset from the train and test folders in the root directory
class HandDiffusionDataset(Dataset):
    def __init__(self, root_dir, channels=1, transform=None, phase='train'):

        self.root_dir = root_dir
        self.transform = transform
        self.channels = channels
       
        self.pth_Image = os.path.join(root_dir, 'jpg')
            
        # file index
        files = [i[:-4] for i in sorted(os.listdir(self.pth_Image))]

        n = len(files)
        train_num = 550
        val_num = 59
        test_num = n - train_num - val_num
        if phase == 'train':
            self.image_files = files[:train_num+val_num]
        elif phase == 'test':
            self.image_files = files[-test_num:]
        elif phase == 'all':
            self.image_files = files
        else:
            raise Exception("Unknown phase: {phase}".format(phase=phase))
        

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_name = self.image_files[idx]

        image = self.read_image(os.path.join(self.pth_Image, image_name + '.jpg'))

        data_dict = {'name': image_name, 'image': image}   

        return data_dict

    def read_image(self, image_path):

        if self.channels == 3:
            image = Image.open(image_path).convert('RGB')
            image_np = np.array(image).astype(np.float32)

        elif self.channels == 1:
            image = Image.open(image_path).convert('L')
            image_np = np.array(image).astype(np.float32)
            image_np = np.expand_dims(image_np, axis=2)
        else:
            raise ValueError('Channels must be either 1 or 3')
        
        if self.transform:
            image = self.transform(image=image_np)['image']
            
        return image


# ------------------------------------------------------------------------
#                               CEPH Dataset
# ------------------------------------------------------------------------

class CephaloDiffusionDataset(Dataset):
    def __init__(self, root_dir, channels=1, transform=None, phase='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.channels = channels
       
        self.pth_Image = os.path.join(root_dir, 'jpg')
            
        # file index
        files = [i[:-4] for i in sorted(os.listdir(self.pth_Image))]

        n = len(files)
        train_num = 130
        val_num = 20
        test_num = n - train_num - val_num
        
        if phase == 'train':
            self.image_files = files[:train_num+val_num]
        elif phase == 'test':
            self.image_files = files[-test_num:]
        elif phase == 'all':
            self.image_files = files
        else:
            raise Exception("Unknown phase: {phase}".format(phase=phase))

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_name = self.image_files[idx]

        image = self.read_image(os.path.join(self.pth_Image, image_name + '.jpg'))

        data_dict = {'name': image_name, 'image': image}   

        return data_dict
    
    def read_image(self, image_path):
            
        if self.channels == 3:
            image = Image.open(image_path).convert('RGB')
            image_np = np.array(image).astype(np.float32)

        elif self.channels == 1:
            image = Image.open(image_path).convert('L')
            image_np = np.array(image).astype(np.float32)
            image_np = np.expand_dims(image_np, axis=2)
        else:
            raise ValueError('Channels must be either 1 or 3')
        
        if self.transform:
            image = self.transform(image=image_np)['image']
            
        return image
    

# ------------------------------------------------------------------------
#                               3D Volume Dataset
# ------------------------------------------------------------------------

class Volume3DDiffusionDataset(Dataset):
    """
    Generic 3D volume dataset for DDPM pretraining.
    Supports NIfTI (.nii, .nii.gz) and NumPy (.npy) formats.
    """
    def __init__(self, root_dir, channels=1, volume_size=64, transform=None, phase='train', file_extension='.nii.gz'):
        self.root_dir = root_dir
        self.transform = transform
        self.channels = channels
        self.volume_size = volume_size
        self.file_extension = file_extension
        
        self.pth_Volume = os.path.join(root_dir, 'volumes')
        
        # file index
        files = [i.replace(file_extension, '') for i in sorted(os.listdir(self.pth_Volume)) if i.endswith(file_extension)]
        
        # Split into train/test (80/20 split by default)
        n = len(files)
        train_num = int(n * 0.8)
        
        if phase == 'train':
            self.volume_files = files[:train_num]
        elif phase == 'test':
            self.volume_files = files[train_num:]
        elif phase == 'all':
            self.volume_files = files
        else:
            raise Exception(f"Unknown phase: {phase}")
    
    def __len__(self):
        return len(self.volume_files)
    
    def __getitem__(self, idx):
        volume_name = self.volume_files[idx]
        volume = self.read_volume(os.path.join(self.pth_Volume, volume_name + self.file_extension))
        data_dict = {'name': volume_name, 'image': volume}
        return data_dict
    
    def read_volume(self, volume_path):
        """
        Read 3D volume from file and process it.
        Supports NIfTI (.nii, .nii.gz) and NumPy (.npy) formats.
        """
        if volume_path.endswith('.npy'):
            # Load NumPy array
            volume_np = np.load(volume_path).astype(np.float32)
        elif volume_path.endswith('.nii') or volume_path.endswith('.nii.gz'):
            # Load NIfTI file
            nii_img = nib.load(volume_path)
            volume_np = nii_img.get_fdata().astype(np.float32)
        else:
            raise ValueError(f'Unsupported file format: {volume_path}')
        
        # Ensure volume is 3D
        if volume_np.ndim == 3:
            # Add channel dimension: (D, H, W) -> (C, D, H, W)
            volume_np = np.expand_dims(volume_np, axis=0)
        elif volume_np.ndim == 4:
            # Assume format is (D, H, W, C) -> transpose to (C, D, H, W)
            volume_np = np.transpose(volume_np, (3, 0, 1, 2))
        else:
            raise ValueError(f'Expected 3D or 4D volume, got shape: {volume_np.shape}')
        
        # Handle channels
        if self.channels == 1:
            if volume_np.shape[0] > 1:
                # Take first channel or average
                volume_np = volume_np[0:1, :, :, :]
        elif self.channels == 3:
            if volume_np.shape[0] == 1:
                # Repeat channel 3 times
                volume_np = np.repeat(volume_np, 3, axis=0)
        
        # Resize volume to target size
        # volume_np shape: (C, D, H, W)
        resized_volume = np.zeros((self.channels, self.volume_size, self.volume_size, self.volume_size), dtype=np.float32)
        for c in range(self.channels):
            resized_volume[c] = resize(
                volume_np[c], 
                (self.volume_size, self.volume_size, self.volume_size),
                mode='constant',
                anti_aliasing=True,
                preserve_range=True
            )
        
        # Normalize to [0, 1]
        min_val = resized_volume.min()
        max_val = resized_volume.max()
        if max_val > min_val:
            resized_volume = (resized_volume - min_val) / (max_val - min_val)
        
        # Convert to torch tensor
        volume_tensor = torch.from_numpy(resized_volume).float()
        
        return volume_tensor
        