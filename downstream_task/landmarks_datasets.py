import numpy as np
import pandas as pd
from PIL import Image
import os

import torch
import utilities
from skimage.transform import resize

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import nibabel as nib  # For 3D medical imaging (NIfTI format)
import SimpleITK as sitk  # For reading .mhd/.raw files
## -----------------------------------------------------------------------------------------------------------------##
##                                                          CHEST DATASET                                           ##
## -----------------------------------------------------------------------------------------------------------------##
"""
LINK: https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels

X-ray images in this data set have been acquired from the tuberculosis control program of the Department of Health and Human Services of Montgomery County, MD, USA. 
This set contains 138 posterior-anterior x-rays, of which 80 x-rays are normal and 58 x-rays are abnormal with manifestations of tuberculosis. 
All images are de-identified and available in DICOM format. The set covers a wide range of abnormalities, including effusions and miliary patterns.
"""
class Chest(torch.utils.data.Dataset):

    def __init__(self, prefix, phase, size=(512, 512), num_channels=1, fuse_heatmap=False, sigma=8):
        self.phase = phase
        self.new_size = size
        self.dataset_name = 'Chest'

        self.transforms = self.get_transforms()
        self.num_channels = num_channels
        self.fuse_heatmap = fuse_heatmap
        self.sigma = sigma
        self.num_landmarks = 6
        self.pth_Image = os.path.join(prefix, 'pngs')
        self.pth_Label = os.path.join(prefix, 'labels')

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
        if self.phase == 'train':
            self.indexes = files[:train_num]
        elif self.phase == 'validate':
            self.indexes = files[train_num:-test_num]
        elif self.phase == 'test':
            self.indexes = files[-test_num:]
        elif self.phase == 'all':
            self.indexes = files
        else:
            raise Exception("Unknown phase: {phase}".format(phase=phase))

    def __getitem__(self, index):
        name = self.indexes[index]
        ret = {'name': name}

        img, img_size= self.readImage(os.path.join(self.pth_Image, name + '.png'))
        points = self.readLandmark(name)        
        heatmaps = utilities.points_to_heatmap(points, sigma=self.sigma, img_size=self.new_size, fuse=self.fuse_heatmap)

        transformed = self.transforms(image=img, masks=heatmaps)
        
        # img shape: CxHxW | heatmaps is a list of CxHxW: example: [CxHxW, CxHxW, CxHxW, CxHxW, CxHxW, CxHxW]
        img, heatmaps = transformed['image'], transformed['masks']
        
        # Image is a torch tensor [C, H, W]
        ret['image'] = img
        ret['landmarks'] = torch.FloatTensor(points)
        # Convert heatmaps to torch tensor [C, H, W]. Stack to give new dimension and float32 type to avoid error in loss function
        ret['heatmaps'] = torch.stack([hm.float() for hm in heatmaps])
        ret['original_size'] = torch.FloatTensor(img_size)
        ret['resized_size'] = torch.FloatTensor(self.new_size)

        return ret

    def __len__(self):
        return len(self.indexes)

    def readLandmark(self, name):
        path = os.path.join(self.pth_Label, name + '.txt')
        points = []
        with open(path, 'r') as f:
            n = int(f.readline())
            for i in range(n):
                ratios = [float(i) for i in f.readline().split()]
                points.append(ratios)
        return np.array(points)
    
    def readImage(self, path):

        if self.num_channels == 3:
            img = Image.open(path).convert('RGB')
            arr = np.array(img).astype(np.float32)
 
        elif self.num_channels == 1:
            img = Image.open(path).convert('L')
            arr = np.array(img).astype(np.float32)
            arr = np.expand_dims(arr, 2)
        else:
            raise ValueError('Channels must be either 1 or 3')

        # Original size in (width, height)
        origin_size = img.size
        resized_image = resize(arr, (self.new_size[0], self.new_size[1], self.num_channels))

        return resized_image, origin_size

    def get_transforms(self):
        if self.phase == 'train':
            return A.Compose([
            A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0, rotate_limit=2, border_mode=cv2.BORDER_REPLICATE, p=0.5),
            #A.Perspective(scale=(0, 0.02), pad_mode=cv2.BORDER_REPLICATE, p=0.5),
            #A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.2, 0.2), p=0.5),
            #A.Resize(self.new_size[0], self.new_size[1]),
            A.Normalize(normalization='min_max'),
            A.pytorch.ToTensorV2()
        ])
        elif self.phase == 'validate':
            return A.Compose([
            #A.Resize(self.new_size[0], self.new_size[1]),
            A.Normalize(normalization='min_max'),
            A.pytorch.ToTensorV2()
            ])
        elif self.phase == 'test':
            return A.Compose([
            #A.Resize(self.new_size[0], self.new_size[1]),
            A.Normalize(normalization='min_max'),
            A.pytorch.transforms.ToTensorV2()
            ])
        else:
            raise ValueError('phase must be either "train" or "validate" or "test"')

## -----------------------------------------------------------------------------------------------------------------##
##                                                          HAND DATASET                                            ##
## -----------------------------------------------------------------------------------------------------------------##

"""
LINK: https://ipilab.usc.edu/research/baaweb/
ASI: Asian; BLK: African American; CAU: Caucasian; HIS: Hispanic.
"""

class Hand(torch.utils.data.Dataset):

    def __init__(self, prefix, phase, size=(512, 368), num_channels=1, fuse_heatmap=False, sigma=5):

        self.phase = phase
        self.new_size = size
        self.dataset_name = 'Hand'
        
        self.transforms = self.get_transforms()
        self.num_channels = num_channels
        self.fuse_heatmap = fuse_heatmap
        self.sigma = sigma
        self.num_landmarks = 37

        self.pth_Image = os.path.join(prefix, 'jpg')
        self.labels = pd.read_csv(os.path.join(
            prefix, 'labels/all.csv'), header=None, index_col=0)

        # file index
        index_set = set(self.labels.index) # Set of all the labels
        files = [i[:-4] for i in sorted(os.listdir(self.pth_Image))] # -4 to cut ".jpg" # List of all the images
        files = [i for i in files if int(i) in index_set] # List of filters that has a label

        n = len(files)
        train_num = 550
        val_num = 59
        test_num = n - train_num - val_num

        if phase == 'train':
            self.indexes = files[:train_num]
        elif phase == 'validate':
            self.indexes = files[train_num:-test_num]
        elif phase == 'test':
            self.indexes = files[-test_num:]
        elif phase == 'all':
            self.indexes = files
        else:
            raise Exception("Unknown phase: {phase}".format(phase=phase))

    def __getitem__(self, index):
        name = self.indexes[index]
        ret = {'name': name}

        img, img_size = self.readImage(
            os.path.join(self.pth_Image, name + '.jpg'))

        points = self.readLandmark(name, img_size)
        heatmaps = utilities.points_to_heatmap(points, sigma=self.sigma, img_size=self.new_size, fuse=self.fuse_heatmap)

        transformed = self.transforms(image=img, masks=heatmaps)
        img, heatmaps = transformed['image'], transformed['masks']
        
        ret['image'] = img
        ret['landmarks'] = torch.FloatTensor(points)
        ret['heatmaps'] = torch.stack([hm.float() for hm in heatmaps])
        ret['original_size'] = torch.FloatTensor(img_size)
        ret['resized_size'] = torch.FloatTensor(self.new_size)

        return ret
    

    def __len__(self):
        return len(self.indexes)

    def readLandmark(self, name, origin_size):
        li = list(self.labels.loc[int(name), :])   
        points = []
        for i in range(0, len(li), 2):
            ratios = (li[i] / origin_size[0], li[i + 1] / origin_size[1])
            points.append(ratios)
        return np.array(points)

    def readImage(self, path):
        
        if self.num_channels == 3:
            img = Image.open(path).convert('RGB')
            arr = np.array(img).astype(np.float32)
 
        elif self.num_channels == 1:
            img = Image.open(path).convert('L')
            arr = np.array(img).astype(np.float32)
            arr = np.expand_dims(arr, 2)
        else:
            raise ValueError('Channels must be either 1 or 3')
            
        # Original size in (width, height)
        origin_size = img.size
        resized_image = resize(arr, (self.new_size[0], self.new_size[1], self.num_channels))

        return resized_image, origin_size
    
    def get_transforms(self):
        if self.phase == 'train':
            return A.Compose([
            A.ShiftScaleRotate(shift_limit=0.02, scale_limit=(-0.02, 0.02), rotate_limit=2, border_mode=cv2.BORDER_REPLICATE, p=0.5),
            #A.Perspective(scale=(0, 0.02), pad_mode=cv2.BORDER_REPLICATE, p=0.5),
            #A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.2, 0.2), p=0.5),
            #A.Resize(self.new_size[0], self.new_size[1]),
            A.Normalize(normalization='min_max'),
            A.pytorch.ToTensorV2()
        ])
        elif self.phase == 'validate':
            return A.Compose([
            #A.Resize(self.new_size[0], self.new_size[1]),
            A.Normalize(normalization='min_max'),
            A.pytorch.ToTensorV2()
            ])
        elif self.phase == 'test':
            return A.Compose([
            #A.Resize(self.new_size[0], self.new_size[1]),
            A.Normalize(normalization='min_max'),
            A.pytorch.transforms.ToTensorV2()
            ])
        else:
            raise ValueError('phase must be either "train" or "validate" or "test"')



## -----------------------------------------------------------------------------------------------------------------##
##                                                          CEPHALOMETRIC DATASET                                            ##
## -----------------------------------------------------------------------------------------------------------------##

"""
LINK: https://www.kaggle.com/datasets/c34a0ef0cd3cfd5c5afbdb30f8541e887171f19f196b1ad63790ca5b28c0ec93
https://figshare.com/s/37ec464af8e81ae6ebbf?file=5466581
"""


class Cephalo(torch.utils.data.Dataset):

    def __init__(self, prefix, phase, size=(512, 416), num_channels=1, fuse_heatmap=False, sigma=5):
        self.phase = phase
        self.new_size = size
        self.dataset_name = 'Cephalo'
        
        self.transforms = self.get_transforms()
        self.num_channels = num_channels
        self.fuse_heatmap = fuse_heatmap
        self.sigma = sigma
        
        self.num_landmarks = 19


        self.pth_Image = os.path.join(prefix, 'jpg')
        self.pth_label_junior = os.path.join(prefix, '400_junior')
        self.pth_label_senior = os.path.join(prefix, '400_senior')

        # file index
        files = [i[:-4] for i in sorted(os.listdir(self.pth_Image))]
        n = len(files)
        
        if phase == 'train':
            self.indexes = files[:130]
        elif phase == 'validate':
            self.indexes = files[130:150]
        elif phase == 'test':
            self.indexes = files[150:400]
        elif phase == 'all':
            self.indexes = files
        else:
            raise Exception("Unknown phase: {phase}".format(phase=phase))


    def __getitem__(self, index):
        name = self.indexes[index]
        ret = {'name': name}

        img, img_size = self.readImage(os.path.join(self.pth_Image, name+'.jpg'))
        points = self.readLandmark(name, img_size)
        heatmaps = utilities.points_to_heatmap(points, sigma=self.sigma, img_size=self.new_size, fuse=self.fuse_heatmap)

        transformed = self.transforms(image=img, masks=heatmaps)        
        img, heatmaps = transformed['image'], transformed['masks']
        
        ret['image'] = img
        ret['landmarks'] = torch.FloatTensor(points)
        ret['heatmaps'] = torch.stack([hm.float() for hm in heatmaps])
        ret['original_size'] = torch.FloatTensor(img_size)
        ret['resized_size'] = torch.FloatTensor(self.new_size)

        return ret

    def __len__(self):
        return len(self.indexes)

    def readLandmark(self, name, origin_size):
        points = []
        with open(os.path.join(self.pth_label_junior, name + '.txt')) as f1:
            with open(os.path.join(self.pth_label_senior, name + '.txt')) as f2:
                for i in range(self.num_landmarks):
                    landmark1 = f1.readline().rstrip('\n').split(',')
                    landmark2 = f2.readline().rstrip('\n').split(',')
                    # Average of junior and senior landmarks
                    landmark = [(float(i) + float(j)) / 2 for i, j in zip(landmark1, landmark2)]
                    #landmark = [float(i) for i in landmark1] 
                    ratios = (landmark[0] / origin_size[0], landmark[1] / origin_size[1])
                    points.append(ratios)
        return np.array(points)

    def readImage(self, path):

        if self.num_channels == 3:
            img = Image.open(path).convert('RGB')
            arr = np.array(img).astype(np.float32)
 
        elif self.num_channels == 1:
            img = Image.open(path).convert('L')
            arr = np.array(img).astype(np.float32)
            arr = np.expand_dims(arr, 2)
        else:
            raise ValueError('Channels must be either 1 or 3')

        # Original size in (width, height)
        origin_size = img.size
        resized_image = resize(arr, (self.new_size[0], self.new_size[1], self.num_channels))
        
        return resized_image, origin_size
    
    def get_transforms(self):
        if self.phase == 'train':
            return A.Compose([
            A.ShiftScaleRotate(shift_limit=0.02, scale_limit=(-0.02, 0.02), rotate_limit=2, border_mode=cv2.BORDER_REPLICATE, p=0.5),
            #A.Perspective(scale=(0, 0.02), pad_mode=cv2.BORDER_REPLICATE, p=0.5),
            #A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.2, 0.2), p=0.5),
            #A.Resize(self.new_size[0], self.new_size[1]),
            A.Normalize(normalization='min_max'),
            A.pytorch.ToTensorV2()
        ])
        elif self.phase == 'validate':
            return A.Compose([
            #A.Resize(self.new_size[0], self.new_size[1]),
            A.Normalize(normalization='min_max'),
            A.pytorch.ToTensorV2()
            ])
        elif self.phase == 'test':
            return A.Compose([
            #A.Resize(self.new_size[0], self.new_size[1]),
            A.Normalize(normalization='min_max'),
            A.pytorch.transforms.ToTensorV2()
            ])
        else:
            raise ValueError('phase must be either "train" or "validate" or "test"')



## -----------------------------------------------------------------------------------------------------------------##
##                                                          3D VOLUME DATASET                                       ##
## -----------------------------------------------------------------------------------------------------------------##

"""
Generic 3D volume dataset for landmark detection.
Supports NIfTI (.nii, .nii.gz) and NumPy (.npy) formats.
"""

class Volume3D(torch.utils.data.Dataset):

    def __init__(self, prefix, phase, size=(64, 64, 64), num_channels=1, fuse_heatmap=False, sigma=2, file_extension='.nii.gz'):
        self.phase = phase
        self.new_size = size
        self.dataset_name = 'Volume3D'
        self.num_channels = num_channels
        self.fuse_heatmap = fuse_heatmap
        self.sigma = sigma
        self.file_extension = file_extension
        
        # Expected directory structure: prefix/volumes/ and prefix/landmarks/
        self.pth_Volume = os.path.join(prefix, 'volumes')
        self.pth_Label = os.path.join(prefix, 'landmarks')
        
        # file index
        files = [i.replace(file_extension, '') for i in sorted(os.listdir(self.pth_Volume)) if i.endswith(file_extension)]
        
        # Split into train/validate/test (80/10/10 split by default)
        n = len(files)
        train_num = int(n * 0.8)
        val_num = int(n * 0.1)
        test_num = n - train_num - val_num
        
        if self.phase == 'train':
            self.indexes = files[:train_num]
        elif self.phase == 'validate':
            self.indexes = files[train_num:train_num+val_num]
        elif self.phase == 'test':
            self.indexes = files[train_num+val_num:]
        elif self.phase == 'all':
            self.indexes = files
        else:
            raise Exception(f"Unknown phase: {phase}")
        
        # Determine number of landmarks from first file
        if len(self.indexes) > 0:
            first_landmark_file = os.path.join(self.pth_Label, self.indexes[0] + '.txt')
            if os.path.exists(first_landmark_file):
                with open(first_landmark_file, 'r') as f:
                    self.num_landmarks = int(f.readline().strip())
            else:
                # Default to a reasonable number if no label file exists
                self.num_landmarks = 10
        else:
            self.num_landmarks = 10

    def __getitem__(self, index):
        name = self.indexes[index]
        ret = {'name': name}

        vol, vol_size = self.readVolume(os.path.join(self.pth_Volume, name + self.file_extension))
        points = self.readLandmark(name)
        
        # Generate 3D heatmaps from landmarks
        heatmaps = utilities.points_to_heatmap_3d(points, sigma=self.sigma, vol_size=self.new_size, fuse=self.fuse_heatmap)
        
        # Volume is already a torch tensor [C, D, H, W]
        ret['image'] = vol
        ret['landmarks'] = torch.FloatTensor(points)
        # Convert heatmaps to torch tensor [C, D, H, W]. Stack to give new dimension and float32 type
        if self.fuse_heatmap:
            ret['heatmaps'] = torch.from_numpy(heatmaps).float().unsqueeze(0)
        else:
            ret['heatmaps'] = torch.from_numpy(heatmaps).float()
        ret['original_size'] = torch.FloatTensor(vol_size)
        ret['resized_size'] = torch.FloatTensor(self.new_size)

        return ret

    def __len__(self):
        return len(self.indexes)

    def readLandmark(self, name):
        """
        Read landmark points from text file.
        Format: First line contains number of landmarks, 
        subsequent lines contain normalized coordinates (x, y, z in range [0, 1])
        """
        path = os.path.join(self.pth_Label, name + '.txt')
        points = []
        with open(path, 'r') as f:
            n = int(f.readline())
            for i in range(n):
                coords = [float(i) for i in f.readline().split()]
                points.append(coords)
        return np.array(points)
    
    def readVolume(self, path):
        """
        Read 3D volume from file and process it.
        Supports NIfTI (.nii, .nii.gz) and NumPy (.npy) formats.
        """
        if path.endswith('.npy'):
            # Load NumPy array
            volume_np = np.load(path).astype(np.float32)
        elif path.endswith('.nii') or path.endswith('.nii.gz'):
            # Load NIfTI file
            nii_img = nib.load(path)
            volume_np = nii_img.get_fdata().astype(np.float32)
        elif path.endswith('.mhd'):
            # Load MetaImage file
            sitk_image = sitk.ReadImage(path)
            volume_np = sitk.GetArrayFromImage(sitk_image).astype(np.float32)
        else:
            raise ValueError(f'Unsupported file format: {path}')
        
        # Store original size
        origin_size = volume_np.shape
        
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
        if self.num_channels == 1:
            if volume_np.shape[0] > 1:
                # Take first channel or average
                volume_np = volume_np[0:1, :, :, :]
        elif self.num_channels == 3:
            if volume_np.shape[0] == 1:
                # Repeat channel 3 times
                volume_np = np.repeat(volume_np, 3, axis=0)
        
        # Resize volume to target size
        # volume_np shape: (C, D, H, W)
        resized_volume = np.zeros((self.num_channels, self.new_size[0], self.new_size[1], self.new_size[2]), dtype=np.float32)
        for c in range(self.num_channels):
            resized_volume[c] = resize(
                volume_np[c], 
                (self.new_size[0], self.new_size[1], self.new_size[2]),
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
        
        return volume_tensor, origin_size
