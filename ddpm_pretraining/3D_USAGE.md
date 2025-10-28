# 3D Data Support for DDPM Pretraining

This document describes how to use the DDPM pretraining framework with 3D volumetric data.

## Overview

The DDPM pretraining framework has been extended to support both 2D images (original functionality) and 3D volumetric data. The implementation maintains backward compatibility with existing 2D workflows while adding comprehensive 3D support.

## Key Features

- **3D Convolutional Networks**: All Conv2d operations have 3D counterparts (Conv3d)
- **3D U-Net Architecture**: Complete 3D U-Net with appropriate downsampling/upsampling
- **3D Attention Mechanisms**: Both standard and linear attention adapted for 3D
- **3D Dataset Loaders**: Support for NIfTI (.nii, .nii.gz) and NumPy (.npy) formats
- **Backward Compatibility**: Original 2D functionality remains unchanged

## Configuration

To enable 3D mode, set the `is_3d` flag in your configuration file:

```json
{
    "is_3d": true,
    "dataset": {
        "name": "volume3d",
        "image_size": 64,
        "image_channels": 1,
        ...
    },
    ...
}
```

### Important Configuration Parameters for 3D

1. **image_size**: Defines the volume size (e.g., 64 means 64×64×64)
2. **image_channels**: Number of channels (typically 1 for medical imaging)
3. **batch_size**: Should be smaller than 2D due to memory constraints (e.g., 1-2)
4. **unet.channel_mults**: May need adjustment for 3D (e.g., [1,2,4] instead of [1,2,4,8])
5. **unet.res_blocks**: Can be reduced for 3D to save memory (e.g., 2 instead of 4)
6. **unet.attn_res**: Should match the feature map size at attention layers (e.g., 16 for 64³ volumes)

## Dataset Preparation

### Directory Structure

For 3D datasets, organize your data as follows:

```
datasets/
└── your_3d_dataset/
    └── volumes/
        ├── volume_001.nii.gz
        ├── volume_002.nii.gz
        ├── volume_003.nii.gz
        └── ...
```

### Supported Formats

1. **NIfTI**: `.nii` or `.nii.gz` (requires nibabel library)
2. **NumPy**: `.npy` arrays

### Data Requirements

- Volumes should be 3D arrays (D, H, W) or 4D (D, H, W, C)
- Data will be automatically resized to the configured `image_size`
- Data will be normalized to [0, 1] range

## Usage Example

### 1. Prepare Your 3D Configuration

Create a configuration file (e.g., `config_3d.json`):

```json
{
    "gpu": 0,
    "experiment_path": "ddpm_pretraining/ddpm_pretraining_experiments",
    "is_3d": true,
    "model": {
        "unet": {
            "channel_mults": [1, 2, 4],
            "attn_res": 16,
            "num_head_channels": 4,
            "res_blocks": 2,
            "self_condition": true
        },
        "beta_schedule": {
            "train": {
                "schedule": "linear",
                "n_timestep": 500,
                "linear_start": 1e-4,
                "linear_end": 0.02
            }
        },
        "lr": 1e-4,
        "optimizer": "adamw",
        "loss_type": "l2",
        "use_ema": true,
        "iterations": 10000,
        "freq_metrics": 1000,
        "freq_checkpoint": 2000
    },
    "dataset": {
        "name": "volume3d",
        "path": "datasets/",
        "image_size": 64,
        "image_channels": 1,
        "batch_size": 2,
        "grad_accumulation": 4,
        "num_workers": 2,
        "pin_memory": true
    }
}
```

### 2. Run Training

```bash
python ddpm_pretraining/main.py --config ddpm_pretraining/config/config_3d.json
```

## Memory Considerations

3D models require significantly more memory than 2D models:

- **2D**: Batch of 4, 256×256 images ≈ 1-2 GB
- **3D**: Batch of 2, 64×64×64 volumes ≈ 4-8 GB

### Tips for Managing Memory

1. **Reduce batch size**: Use 1-2 instead of 4-8
2. **Increase gradient accumulation**: Compensate for smaller batches
3. **Reduce volume size**: Use 32×32×32 or 48×48×48 instead of 64×64×64
4. **Simplify model**: Use fewer channel multipliers [1,2,4] instead of [1,2,4,8]
5. **Reduce attention**: Lower attention resolution or use fewer attention heads

## Custom 3D Dataset

If you need to create a custom 3D dataset, extend the `Volume3DDiffusionDataset` class:

```python
from ddpm_datasets import Volume3DDiffusionDataset

class MyCustom3DDataset(Volume3DDiffusionDataset):
    def read_volume(self, volume_path):
        # Custom loading logic
        volume = your_custom_loader(volume_path)
        
        # Ensure correct shape (C, D, H, W)
        # Normalize to [0, 1]
        # Return as torch tensor
        return volume_tensor
```

## Differences from 2D

### Architecture Changes

1. **Convolutions**: Conv2d → Conv3d
2. **Downsampling**: Rearrange with p1=2, p2=2 → p1=2, p2=2, p3=2
3. **Upsampling**: 2D nearest neighbor → 3D nearest neighbor
4. **Attention**: Spatial dimensions (H, W) → (D, H, W)

### Input/Output Shapes

- **2D Input**: (B, C, H, W)
- **3D Input**: (B, C, D, H, W)

### Noise Schedule

The noise schedule remains the same for both 2D and 3D, but the noise tensor shape differs:

- **2D**: `(B, C, H, W)`
- **3D**: `(B, C, D, H, W)`

## Testing

To verify your 3D setup is working:

```python
import torch
from model.ddpm_model import DDPM

# Create 3D DDPM model
ddpm = DDPM(
    image_size=32,
    channels=1,
    device="cuda",
    is_3d=True
)

# Test forward pass
batch_size = 2
x = torch.randn(batch_size, 1, 32, 32, 32).cuda()
t = ddpm.sample_timesteps(batch_size)
loss = ddpm.p_losses(x, t)

print(f"Loss: {loss.item()}")
```

## Troubleshooting

### Out of Memory

- Reduce batch size to 1
- Reduce volume size (e.g., 32×32×32)
- Reduce model complexity (fewer channels, blocks)
- Enable gradient checkpointing (already enabled in forward pass)

### Slow Training

- Use fewer timesteps (e.g., 200 instead of 500)
- Reduce attention resolution
- Use fewer residual blocks
- Optimize num_workers for your system

### Data Loading Issues

- Ensure nibabel is installed: `pip install nibabel`
- Check volume file format (.nii, .nii.gz, or .npy)
- Verify volume dimensions are correct
- Check file permissions

## Example Datasets

The following 3D medical imaging datasets are compatible:

- **LUNA16**: Lung nodule detection (CT scans)
- **BraTS**: Brain tumor segmentation (MRI)
- **Medical Segmentation Decathlon**: Various organs
- **KITS**: Kidney tumor segmentation

Remember to preprocess data to the expected format and structure.
