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
from skimage.transform import resize
import nibabel as nib  # For 3D medical imaging (NIfTI format)
import SimpleITK as sitk  # For reading .mhd/.raw files

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


# ------------------------------------------------------------------------
#                               LUNA16 Dataset (with MONAI)
# ------------------------------------------------------------------------

class LUNA16DiffusionDataset(Dataset):
    """
    LUNA16 CT scan dataset for DDPM pretraining.
    Supports MetaImage (.mhd/.raw) format with GPU-accelerated transforms.
    """
    def __init__(self, root_dir, channels=1, volume_size=64, transform=None, phase='train', subset_ids=None, use_gpu_transforms=True):
        """
        Args:
            root_dir: Root directory containing subset folders (subset0-subset9)
            channels: Number of channels (1 for grayscale CT)
            volume_size: Target volume size for resizing
            transform: Optional transforms
            phase: 'train', 'test', or 'all'
            subset_ids: List of subset IDs to use (e.g., [0,1,2,3,4] for 5-fold CV)
                       If None, will auto-split based on phase
            use_gpu_transforms: Whether to use GPU-accelerated MONAI transforms
        """
        self.root_dir = root_dir
        self.transform = transform
        self.channels = channels
        self.volume_size = volume_size
        self.use_gpu_transforms = use_gpu_transforms
        
        # Import MONAI transforms if using GPU acceleration
        if self.use_gpu_transforms:
            try:
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
                        spatial_size=(volume_size, volume_size, volume_size),
                        mode='trilinear',  # Better quality than nearest
                        align_corners=True
                    ),
                    ToTensor()
                ])
                print("LUNA16Dataset: Using GPU-accelerated MONAI transforms")
            except ImportError:
                print("Warning: MONAI not found. Install with: pip install monai")
                print("Falling back to CPU-based transforms")
                self.use_gpu_transforms = False
        
        # Collect all .mhd files from subsets
        all_files = []
        for subset_id in range(10):  # subset0 to subset9
            subset_dir = os.path.join(root_dir, f'subset{subset_id}', f'subset{subset_id}')
            if os.path.exists(subset_dir):
                mhd_files = [f for f in os.listdir(subset_dir) if f.endswith('.mhd')]
                for mhd_file in mhd_files:
                    all_files.append({
                        'subset': subset_id,
                        'filename': mhd_file,
                        'path': os.path.join(subset_dir, mhd_file)
                    })
        
        # Sort by filename for reproducibility
        all_files = sorted(all_files, key=lambda x: x['filename'])
        
        # Split based on subset_ids or phase
        if subset_ids is not None:
            # Use specific subsets
            self.volume_files = [f for f in all_files if f['subset'] in subset_ids]
        else:
            # Default 10-fold split: use first 8 subsets for train, last 2 for test
            train_subsets = list(range(8))  # subsets 0-7 for training
            test_subsets = list(range(8, 10))  # subsets 8-9 for testing
            
            if phase == 'train':
                self.volume_files = [f for f in all_files if f['subset'] in train_subsets]
            elif phase == 'test':
                self.volume_files = [f for f in all_files if f['subset'] in test_subsets]
            elif phase == 'all':
                self.volume_files = all_files
            else:
                raise Exception(f"Unknown phase: {phase}")
        
        print(f"LUNA16Dataset [{phase}]: Loaded {len(self.volume_files)} CT scans")
    
    def __len__(self):
        return len(self.volume_files)
    
    def __getitem__(self, idx):
        file_info = self.volume_files[idx]
        volume = self.read_volume(file_info['path'])
        
        # Use SeriesInstanceUID (filename without extension) as name
        volume_name = file_info['filename'].replace('.mhd', '')
        
        data_dict = {'name': volume_name, 'image': volume}
        return data_dict
    
    def read_volume(self, volume_path):
        """
        Read 3D CT volume from MetaImage (.mhd/.raw) format.
        Uses GPU-accelerated MONAI transforms if available.
        """
        # Read .mhd file using SimpleITK
        sitk_image = sitk.ReadImage(volume_path)
        volume_np = sitk.GetArrayFromImage(sitk_image).astype(np.float32)
        
        if self.use_gpu_transforms:
            # Use MONAI transforms (GPU-accelerated)
            # MONAI expects (C, D, H, W) or (D, H, W)
            volume_tensor = self.monai_transforms(volume_np)
            
            # Ensure correct number of channels
            if volume_tensor.shape[0] != self.channels:
                if self.channels == 1 and volume_tensor.shape[0] > 1:
                    volume_tensor = volume_tensor[0:1]
                elif self.channels == 3 and volume_tensor.shape[0] == 1:
                    volume_tensor = volume_tensor.repeat(3, 1, 1, 1)
        else:
            # Fallback to CPU-based transforms (original method)
            # SimpleITK returns (D, H, W) format
            # Add channel dimension: (D, H, W) -> (1, D, H, W)
            volume_np = np.expand_dims(volume_np, axis=0)
            
            # Clip HU values (typical CT range: -1000 to 400 for lung)
            volume_np = np.clip(volume_np, -1000, 400)
            
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
