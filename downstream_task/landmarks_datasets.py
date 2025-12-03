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
        Read 3D CT volume from MetaImage (.mhd) file using GPU-accelerated MONAI transforms.
        
        Returns:
            volume_tensor: Processed volume as torch tensor [C, D, H, W]
            origin_size: Original volume dimensions
            origin: World coordinate origin
            spacing: Voxel spacing in mm
        """
        # Read MetaImage file using SimpleITK
        sitk_image = sitk.ReadImage(path)
        
        # Get metadata
        origin = sitk_image.GetOrigin()      # World coordinate origin
        spacing = sitk_image.GetSpacing()    # Voxel spacing in mm
        
        # Get volume as numpy array
        # SimpleITK returns (z, y, x) ordering which is (D, H, W)
        volume_np = sitk.GetArrayFromImage(sitk_image).astype(np.float32)
        
        # Store original size (D, H, W)
        origin_size = volume_np.shape
        
        # Apply MONAI transforms (GPU-accelerated)
        # MONAI expects (D, H, W) or (C, D, H, W)
        volume_tensor = self.monai_transforms(volume_np)
        
        # Ensure correct number of channels
        if volume_tensor.shape[0] != self.num_channels:
            if self.num_channels == 1 and volume_tensor.shape[0] > 1:
                volume_tensor = volume_tensor[0:1]
            elif self.num_channels == 3 and volume_tensor.shape[0] == 1:
                volume_tensor = volume_tensor.repeat(3, 1, 1, 1)
        
        return volume_tensor, origin_size, origin, spacing

    def apply_ct_window(self, volume, window_center=-600, window_width=1500):
        """
        Apply CT windowing to enhance relevant structures.
        
        Args:
            volume: Raw CT volume in Hounsfield Units
            window_center: Window center (default -600 for lung)
            window_width: Window width (default 1500 for lung)
        
        Returns:
            Windowed volume normalized to [0, 1]
        """
        min_val = window_center - window_width // 2
        max_val = window_center + window_width // 2
        
        volume = np.clip(volume, min_val, max_val)
        volume = (volume - min_val) / (max_val - min_val)
        
        return volume

class LUNA16(torch.utils.data.Dataset):

    def __init__(self, prefix, phase, size=(64, 64, 64), num_channels=1, fuse_heatmap=False, sigma=2, 
                 subsets=None, max_landmarks=None):
        """
        Initialize LUNA16 dataset for landmark detection.
        
        Args:
            prefix: Path to LUNA16 dataset root directory
            phase: 'train', 'validate', 'test', or 'all'
            size: Target volume size (D, H, W)
            num_channels: Number of input channels (typically 1 for CT)
            fuse_heatmap: Whether to fuse all heatmaps into one
            sigma: Standard deviation for 3D Gaussian heatmaps
            subsets: List of subset indices to use (e.g., [0,1,2,3,4,5,6] for train)
                    If None, uses default 10-fold cross-validation split
            max_landmarks: Maximum number of landmarks per scan (pads/truncates if set)
        """
        self.phase = phase
        self.new_size = size
        self.dataset_name = 'LUNA16'
        self.num_channels = num_channels
        self.fuse_heatmap = fuse_heatmap
        self.sigma = sigma
        self.prefix = prefix
        self.max_landmarks = max_landmarks
        
        # Import MONAI transforms for GPU acceleration
        from monai.transforms import (
            Compose, 
            Resize, 
            ScaleIntensityRange,
            EnsureChannelFirst,
            ToTensor
        )
        
        # Create MONAI transform pipeline
        self.monai_transforms = Compose([
            EnsureChannelFirst(channel_dim='no_channel'),  # Adds channel dim if needed
            ScaleIntensityRange(
                a_min=-1000.0,  # HU min for lung
                a_max=400.0,    # HU max for lung
                b_min=0.0,
                b_max=1.0,
                clip=True
            ),
            Resize(
                spatial_size=(size[0], size[1], size[2]),
                mode='trilinear',  # Better quality than nearest
                align_corners=True
            ),
            ToTensor()
        ])
        print("LUNA16 Landmark Dataset: Using GPU-accelerated MONAI transforms")
        
        # Load annotations
        self.annotations_df = pd.read_csv(os.path.join(prefix, 'annotations.csv'))
        
        # Group annotations by SeriesInstanceUID
        self.annotations_grouped = self.annotations_df.groupby('seriesuid')
        
        # Determine which subsets to use based on phase
        if subsets is not None:
            self.subsets = subsets
        else:
            if phase == 'train':
                self.subsets = [5,6,7]
            elif phase == 'validate':
                self.subsets = [8]
            elif phase == 'test':
                self.subsets = [9]
            elif phase == 'all':
                self.subsets = [0,1,2,3,4,5,6,7,8,9]
            else:
                raise Exception(f"Unknown phase: {phase}")
        
        # Build list of available scans
        self.scan_paths = {}  # uid -> path to .mhd file
        self.indexes = []
        total_scans_found = 0
        skipped_no_landmarks = 0
        
        for subset_idx in self.subsets:
            # Try both directory structures (some LUNA16 downloads have different structures)
            subset_dir_options = [
                os.path.join(prefix, f'subset{subset_idx}', f'subset{subset_idx}'),  # Nested structure
                os.path.join(prefix, f'subset{subset_idx}'),  # Flat structure
            ]
            
            subset_dir = None
            for option in subset_dir_options:
                if os.path.exists(option):
                    subset_dir = option
                    break
            
            if subset_dir is None:
                print(f"Warning: subset{subset_idx} not found in {prefix}")
                continue
            
            for filename in os.listdir(subset_dir):
                if filename.endswith('.mhd'):
                    uid = filename.replace('.mhd', '')
                    total_scans_found += 1
                    
                    # Only include scans that have annotations with at least 1 landmark
                    if uid in self.annotations_grouped.groups:
                        num_annotations = len(self.annotations_grouped.get_group(uid))
                        if num_annotations > 0:
                            self.scan_paths[uid] = os.path.join(subset_dir, filename)
                            self.indexes.append(uid)
                        else:
                            skipped_no_landmarks += 1
                    else:
                        skipped_no_landmarks += 1
        
        # Sort for reproducibility
        self.indexes = sorted(self.indexes)
        
        # Determine number of landmarks (max nodules per scan or fixed)
        if self.max_landmarks is None:
            # Find the maximum number of nodules in any scan
            self.num_landmarks = self.annotations_grouped.size().max()
        else:
            self.num_landmarks = self.max_landmarks
        
        print(f"LUNA16 {phase} dataset: {len(self.indexes)} scans with landmarks, max {self.num_landmarks} landmarks per scan")
        print(f"Total scans found: {total_scans_found}, Skipped (0 landmarks): {skipped_no_landmarks}")
        
        # Add volume cache to speed up data loading
        self.volume_cache = {}
        self.use_cache = True  # Set to False if memory is an issue

    def __getitem__(self, index):
        uid = self.indexes[index]
        ret = {'name': uid}

        # Read CT volume (with caching)
        if self.use_cache and uid in self.volume_cache:
            vol, vol_size, origin, spacing = self.volume_cache[uid]
        else:
            vol, vol_size, origin, spacing = self.readVolume(self.scan_paths[uid])
            if self.use_cache:
                self.volume_cache[uid] = (vol, vol_size, origin, spacing)
        
        # Read landmarks (nodule positions) - normalized [0,1]
        points_normalized, num_valid_landmarks = self.readLandmarks(uid, vol_size, origin, spacing)
        
        # Only use valid landmarks for heatmap generation (exclude padding)
        valid_points_normalized = points_normalized[:num_valid_landmarks]
        
        # Scale normalized points to NEW resized volume coordinates
        points_resized = valid_points_normalized * np.array(self.new_size)
        
        # Generate 3D heatmaps using resized coordinates - ONLY for valid landmarks
        if num_valid_landmarks > 0:
            heatmaps = utilities.points_to_heatmap_3d(
                points_resized,  # Only valid points, already scaled
                sigma=self.sigma, 
                vol_size=self.new_size,
                fuse=self.fuse_heatmap
            )
        else:
            # No valid landmarks - create empty heatmaps
            if self.fuse_heatmap:
                heatmaps = np.zeros(self.new_size, dtype=np.float32)
            else:
                heatmaps = np.zeros((self.num_landmarks, *self.new_size), dtype=np.float32)
        
        # Pad heatmaps to fixed number if not fused and we have fewer valid landmarks
        if not self.fuse_heatmap and num_valid_landmarks < self.num_landmarks:
            padding_shape = (self.num_landmarks - num_valid_landmarks, *self.new_size)
            padding = np.zeros(padding_shape, dtype=np.float32)
            heatmaps = np.concatenate([heatmaps, padding], axis=0)
        
        # Volume is already a torch tensor [C, D, H, W]
        ret['image'] = vol
        ret['landmarks'] = torch.FloatTensor(points_normalized)  # Full array with padding
        ret['num_valid_landmarks'] = num_valid_landmarks
        
        # Convert heatmaps to torch tensor
        if self.fuse_heatmap:
            ret['heatmaps'] = torch.from_numpy(heatmaps).float().unsqueeze(0)
        else:
            ret['heatmaps'] = torch.from_numpy(heatmaps).float()
        
        ret['original_size'] = torch.FloatTensor(vol_size)
        ret['resized_size'] = torch.FloatTensor(self.new_size)
        ret['origin'] = torch.FloatTensor(origin)
        ret['spacing'] = torch.FloatTensor(spacing)

        return ret

    def __len__(self):
        return len(self.indexes)

    def world_to_voxel(self, world_coord, origin, spacing):
        """
        Convert world coordinates to voxel coordinates.
        
        Args:
            world_coord: (x, y, z) in world space (mm)
            origin: Volume origin in world space
            spacing: Voxel spacing in mm
        
        Returns:
            Voxel coordinates (z, y, x) - note the order change for numpy indexing
        """
        # LUNA16 uses (x, y, z) world coordinates
        # Convert to voxel indices
        voxel_coord = (np.array(world_coord) - np.array(origin)) / np.array(spacing)
        # Return in (z, y, x) order for numpy array indexing
        return [voxel_coord[2], voxel_coord[1], voxel_coord[0]]

    def readLandmarks(self, uid, vol_size, origin, spacing):
        """
        Read nodule annotations for a given scan and convert to normalized coordinates.
        
        Args:
            uid: SeriesInstanceUID of the scan
            vol_size: Original volume size (D, H, W)
            origin: Volume origin in world coordinates
            spacing: Voxel spacing
        
        Returns:
            points: Array of normalized landmark coordinates (N, 3) in (z, y, x) order
            num_valid: Number of valid (non-padded) landmarks
        """
        # Get annotations for this scan
        scan_annotations = self.annotations_grouped.get_group(uid)
        
        points = []
        for _, row in scan_annotations.iterrows():
            # World coordinates from annotation
            world_coord = [row['coordX'], row['coordY'], row['coordZ']]
            
            # Convert to voxel coordinates (z, y, x)
            voxel_coord = self.world_to_voxel(world_coord, origin, spacing)
            
            # Normalize to [0, 1] range
            normalized_coord = [
                voxel_coord[0] / vol_size[0],  # z / D
                voxel_coord[1] / vol_size[1],  # y / H
                voxel_coord[2] / vol_size[2],  # x / W
            ]
            
            # Clip to valid range
            normalized_coord = [max(0, min(1, c)) for c in normalized_coord]
            points.append(normalized_coord)
        
        num_valid = len(points)
        
        # Pad or truncate to fixed number of landmarks
        if len(points) < self.num_landmarks:
            # Pad with -1 (invalid marker) instead of [0,0,0]
            while len(points) < self.num_landmarks:
                points.append([-1.0, -1.0, -1.0])  # Invalid marker - won't create corner heatmaps
        elif len(points) > self.num_landmarks:
            # Truncate
            points = points[:self.num_landmarks]
            num_valid = self.num_landmarks
        
        return np.array(points), num_valid

    def readVolume(self, path):
        """
        Read 3D CT volume from MetaImage (.mhd) file using GPU-accelerated MONAI transforms.
        
        Returns:
            volume_tensor: Processed volume as torch tensor [C, D, H, W]
            origin_size: Original volume dimensions
            origin: World coordinate origin
            spacing: Voxel spacing in mm
        """
        # Read MetaImage file using SimpleITK
        sitk_image = sitk.ReadImage(path)
        
        # Get metadata
        origin = sitk_image.GetOrigin()      # World coordinate origin
        spacing = sitk_image.GetSpacing()    # Voxel spacing in mm
        
        # Get volume as numpy array
        # SimpleITK returns (z, y, x) ordering which is (D, H, W)
        volume_np = sitk.GetArrayFromImage(sitk_image).astype(np.float32)
        
        # Store original size (D, H, W)
        origin_size = volume_np.shape
        
        # Apply MONAI transforms (GPU-accelerated)
        # MONAI expects (D, H, W) or (C, D, H, W)
        volume_tensor = self.monai_transforms(volume_np)
        
        # Ensure correct number of channels
        if volume_tensor.shape[0] != self.num_channels:
            if self.num_channels == 1 and volume_tensor.shape[0] > 1:
                volume_tensor = volume_tensor[0:1]
            elif self.num_channels == 3 and volume_tensor.shape[0] == 1:
                volume_tensor = volume_tensor.repeat(3, 1, 1, 1)
        
        return volume_tensor, origin_size, origin, spacing