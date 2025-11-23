# 3D Data Support for Downstream Task (Landmark Detection)

This document describes how to use the downstream task module with 3D volumetric data for landmark detection.

## Overview

The downstream task module has been extended to support both 2D images (original functionality) and 3D volumetric data. The implementation maintains backward compatibility with existing 2D workflows while adding comprehensive 3D support for medical imaging landmark detection.

## Key Features

- **3D Convolutional Networks**: All Conv2d operations have 3D counterparts (Conv3d)
- **3D U-Net Architecture**: Complete 3D U-Net with appropriate downsampling/upsampling
- **3D Attention Mechanisms**: Both standard and linear attention adapted for 3D
- **3D Dataset Loaders**: Support for NIfTI (.nii, .nii.gz), NumPy (.npy), and MetaImage (.mhd) formats
- **3D Heatmap Generation**: 3D Gaussian heatmaps for landmark localization
- **3D Landmark Extraction**: Extract 3D landmark coordinates from volumetric heatmaps
- **Backward Compatibility**: Original 2D functionality remains unchanged

## Configuration

To enable 3D mode, set the `is_3d` flag in your configuration file:

```json
{
    "is_3d": true,
    "dataset": {
        "name": "volume3d",
        "image_size": [64, 64, 64],
        "image_channels": 1,
        "sigma": 2,
        "batch_size": 1,
        ...
    },
    ...
}
```

### Important Configuration Parameters for 3D

1. **image_size**: Defines the volume size as [D, H, W] (e.g., [64, 64, 64])
2. **image_channels**: Number of channels (typically 1 for medical imaging)
3. **batch_size**: Should be smaller than 2D due to memory constraints (e.g., 1-2)
4. **grad_accumulation**: Should be higher to compensate for smaller batch size (e.g., 16)
5. **sigma**: Standard deviation for 3D Gaussian heatmaps (e.g., 2-3)

## Dataset Preparation

### Directory Structure

For 3D datasets, organize your data as follows:

```
datasets/
└── your_3d_dataset/
    ├── volumes/
    │   ├── volume_001.nii.gz
    │   ├── volume_002.nii.gz
    │   ├── volume_003.nii.gz
    │   └── ...
    └── landmarks/
        ├── volume_001.txt
        ├── volume_002.txt
        ├── volume_003.txt
        └── ...
```

### Landmark File Format

Each landmark file should contain:
- First line: Number of landmarks (integer)
- Subsequent lines: Normalized coordinates (x, y, z) in range [0, 1], space-separated

Example `volume_001.txt`:
```
10
0.234 0.456 0.678
0.345 0.567 0.789
0.123 0.234 0.345
...
```

### Supported Formats

1. **NIfTI**: `.nii` or `.nii.gz` (requires nibabel library)
2. **NumPy**: `.npy` arrays
3. **MetaImage**: `.mhd` (requires SimpleITK library)

### Data Requirements

- Volumes should be 3D arrays (D, H, W) or 4D (D, H, W, C)
- Data will be automatically resized to the configured `image_size`
- Data will be normalized to [0, 1] range
- Landmarks should be normalized to [0, 1] range

## Usage Example

### 1. Prepare Your 3D Configuration

Use the provided configuration template (`config/config_3d.json`) or create your own:

```json
{
    "gpu": 0,
    "experiment_path": "downstream_task/landmarks_experiments_3d",
    "is_3d": true,
    "model": {
        "name": "ddpm",
        "lr": 1e-5,
        "optimizer": "AdamW",
        "epochs": 200
    },
    "dataset": {
        "name": "volume3d",
        "path": "datasets/",
        "image_size": [64, 64, 64],
        "image_channels": 1,
        "sigma": 2,
        "batch_size": 1,
        "grad_accumulation": 16
    }
}
```

### 2. Run Training

```bash
python downstream_task/main.py --config downstream_task/config/config_3d.json
```

### 3. Run Inference

Inference works the same way as 2D, but with 3D volumes as input.

## Memory Considerations

3D models require significantly more memory than 2D models:

- **2D**: Batch of 2, 256×256 images ≈ 0.5-1 GB
- **3D**: Batch of 1, 64×64×64 volumes ≈ 4-6 GB

### Tips for Managing Memory

1. **Reduce batch size**: Use 1 instead of 2-4
2. **Increase gradient accumulation**: Compensate for smaller batches (e.g., 16 instead of 8)
3. **Reduce volume size**: Use 48×48×48 or 32×32×32 instead of 64×64×64
4. **Simplify model**: Use fewer channel multipliers [1,2,4] instead of [1,2,4,8]
5. **Reduce sigma**: Use sigma=2 instead of sigma=5 for smaller heatmaps

## Model Architecture

The 3D U-Net architecture is similar to 2D but with 3D convolutions:

- **Input**: (Batch, Channels, Depth, Height, Width)
- **Output**: (Batch, Num_Landmarks, Depth, Height, Width) for heatmaps
- **3D Convolutions**: All Conv2d replaced with Conv3d when is_3d=True
- **3D Attention**: Spatial attention adapted for 3D volumes
- **3D Pooling**: Downsampling and upsampling in 3D

## Differences from 2D

### Architecture Changes

1. **Convolutions**: Conv2d → Conv3d
2. **Downsampling**: Rearrange from (H, W) → (H/2, W/2) to (D, H, W) → (D/2, H/2, W/2)
3. **Upsampling**: 2D nearest neighbor → 3D nearest neighbor
4. **Attention**: Spatial dimensions (H, W) → (D, H, W)

### Input/Output Shapes

- **2D Input**: (B, C, H, W)
- **3D Input**: (B, C, D, H, W)
- **2D Heatmaps**: (B, N, H, W)
- **3D Heatmaps**: (B, N, D, H, W)
- **2D Landmarks**: (N, 2) where N is number of landmarks
- **3D Landmarks**: (N, 3) where N is number of landmarks

### Metrics

All metrics (MSE, MRE, SDR, mAP) work with both 2D and 3D as they operate on landmark coordinates.

## Testing

To verify your 3D setup is working, you can create a simple test:

```python
import torch
from model.models import Unet

# Create 3D U-Net model
model = Unet(
    dim=64,
    channels=1,
    dim_mults=[1, 2, 4],
    self_condition=False,
    resnet_block_groups=4,
    att_heads=4,
    att_res=16,
    is_3d=True
)

# Test forward pass
batch_size = 1
x = torch.randn(batch_size, 1, 64, 64, 64)
output = model(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
```

## Troubleshooting

### Out of Memory

- Reduce batch size to 1
- Reduce volume size (e.g., 32×32×32 or 48×48×48)
- Reduce model complexity (fewer channels, blocks)
- Increase gradient accumulation

### Slow Training

- Reduce volume size
- Use fewer residual blocks
- Reduce attention resolution
- Optimize num_workers for your system

### Data Loading Issues

- Ensure nibabel and SimpleITK are installed: `pip install nibabel simpleitk`
- Check volume file format (.nii, .nii.gz, .mhd, or .npy)
- Verify volume dimensions are correct
- Check file permissions
- Ensure landmark files match volume files

### Landmark Detection Issues

- Adjust sigma parameter for better heatmap coverage
- Check landmark normalization (should be in [0, 1])
- Verify landmark file format (first line = number of landmarks)
- Ensure volume and landmark files are aligned

## Example Datasets

The following 3D medical imaging datasets can be adapted for landmark detection:

- **Spine CT**: Vertebrae landmark detection
- **Brain MRI**: Anatomical landmark detection
- **Cardiac CT/MRI**: Cardiac landmark detection
- **Chest CT**: Lung nodule/lesion landmark detection

Remember to preprocess data to the expected format and structure, and create appropriate landmark annotations.

## Best Practices

1. **Start Small**: Begin with smaller volumes (32³ or 48³) for faster iteration
2. **Validate Data**: Always visualize a few samples to ensure correct loading
3. **Monitor Memory**: Use `nvidia-smi` to monitor GPU memory usage
4. **Adjust Sigma**: Tune sigma parameter based on your landmark spacing
5. **Use Pretrained Weights**: If available, use pretrained 3D DDPM weights
6. **Augmentation**: Consider adding 3D data augmentation in the future

## Limitations

Current implementation does not include:
- 3D data augmentation (rotations, flips, etc.)
- Anisotropic volume support (different resolution per axis)
- Patch-based training for larger volumes
- 3D visualization utilities

These features may be added in future updates.

## Summary

The downstream task module now supports 3D volumetric landmark detection while maintaining 100% backward compatibility with existing 2D workflows. The changes follow the same patterns as the ddpm_pretraining module and are well-documented for easy usage.
