import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import torch
from mpl_toolkits.mplot3d import Axes3D
import os

# Import your dataset
from landmarks_datasets import LUNA16, LUNA16Candidates


def visualize_slices_with_landmarks(volume, landmarks, heatmaps, name, save_dir='visualizations'):
    """
    Visualize axial, sagittal, and coronal slices with landmarks overlaid.
    
    Args:
        volume: torch tensor [C, D, H, W]
        landmarks: torch tensor [N, 3] - normalized coordinates (z, y, x)
        heatmaps: torch tensor [N, D, H, W] or [1, D, H, W] if fused
        name: scan identifier
        save_dir: directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert to numpy
    vol = volume.squeeze().numpy()  # [D, H, W]
    lm = landmarks.numpy()
    hm = heatmaps.numpy()
    
    D, H, W = vol.shape
    
    # Denormalize landmark coordinates
    lm_voxel = lm.copy()
    lm_voxel[:, 0] *= D  # z
    lm_voxel[:, 1] *= H  # y
    lm_voxel[:, 2] *= W  # x
    
    # Filter out padding landmarks (at origin)
    valid_mask = ~np.all(lm == 0, axis=1)
    lm_voxel_valid = lm_voxel[valid_mask]
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(f'LUNA16 Scan: {name}\nVolume shape: {vol.shape}, Landmarks: {valid_mask.sum()}', fontsize=14)
    
    # For each valid landmark, show slices through it
    if len(lm_voxel_valid) > 0:
        # Use first landmark as reference
        z_ref, y_ref, x_ref = lm_voxel_valid[0].astype(int)
        z_ref = np.clip(z_ref, 0, D-1)
        y_ref = np.clip(y_ref, 0, H-1)
        x_ref = np.clip(x_ref, 0, W-1)
    else:
        z_ref, y_ref, x_ref = D//2, H//2, W//2
    
    # Row 1: Volume slices
    # Axial slice (z)
    axes[0, 0].imshow(vol[z_ref, :, :], cmap='gray')
    axes[0, 0].set_title(f'Axial (z={z_ref})')
    for lm_pt in lm_voxel_valid:
        if abs(lm_pt[0] - z_ref) < 3:  # Show landmarks within 3 slices
            axes[0, 0].scatter(lm_pt[2], lm_pt[1], c='red', s=100, marker='x', linewidths=2)
    
    # Coronal slice (y)
    axes[0, 1].imshow(vol[:, y_ref, :], cmap='gray', aspect='auto')
    axes[0, 1].set_title(f'Coronal (y={y_ref})')
    for lm_pt in lm_voxel_valid:
        if abs(lm_pt[1] - y_ref) < 3:
            axes[0, 1].scatter(lm_pt[2], lm_pt[0], c='red', s=100, marker='x', linewidths=2)
    
    # Sagittal slice (x)
    axes[0, 2].imshow(vol[:, :, x_ref], cmap='gray', aspect='auto')
    axes[0, 2].set_title(f'Sagittal (x={x_ref})')
    for lm_pt in lm_voxel_valid:
        if abs(lm_pt[2] - x_ref) < 3:
            axes[0, 2].scatter(lm_pt[1], lm_pt[0], c='red', s=100, marker='x', linewidths=2)
    
    # Volume histogram
    axes[0, 3].hist(vol.flatten(), bins=50, color='blue', alpha=0.7)
    axes[0, 3].set_title('Volume Intensity Distribution')
    axes[0, 3].set_xlabel('Intensity')
    axes[0, 3].set_ylabel('Count')
    
    # Row 2: Heatmap slices (sum or max projection)
    if hm.ndim == 4:
        # Multiple heatmaps [N, D, H, W]
        hm_combined = np.max(hm, axis=0)  # Max across landmarks
    else:
        hm_combined = hm
    
    # Axial heatmap
    axes[1, 0].imshow(hm_combined[z_ref, :, :], cmap='hot')
    axes[1, 0].set_title(f'Heatmap Axial (z={z_ref})')
    
    # Coronal heatmap
    axes[1, 1].imshow(hm_combined[:, y_ref, :], cmap='hot', aspect='auto')
    axes[1, 1].set_title(f'Heatmap Coronal (y={y_ref})')
    
    # Sagittal heatmap
    axes[1, 2].imshow(hm_combined[:, :, x_ref], cmap='hot', aspect='auto')
    axes[1, 2].set_title(f'Heatmap Sagittal (x={x_ref})')
    
    # Heatmap histogram
    axes[1, 3].hist(hm_combined.flatten(), bins=50, color='red', alpha=0.7)
    axes[1, 3].set_title('Heatmap Intensity Distribution')
    axes[1, 3].set_xlabel('Intensity')
    axes[1, 3].set_ylabel('Count')
    
    # Row 3: Overlay (volume + heatmap)
    alpha = 0.5
    
    # Axial overlay
    axes[2, 0].imshow(vol[z_ref, :, :], cmap='gray')
    axes[2, 0].imshow(hm_combined[z_ref, :, :], cmap='hot', alpha=alpha)
    axes[2, 0].set_title(f'Overlay Axial (z={z_ref})')
    
    # Coronal overlay
    axes[2, 1].imshow(vol[:, y_ref, :], cmap='gray', aspect='auto')
    axes[2, 1].imshow(hm_combined[:, y_ref, :], cmap='hot', alpha=alpha, aspect='auto')
    axes[2, 1].set_title(f'Overlay Coronal (y={y_ref})')
    
    # Sagittal overlay
    axes[2, 2].imshow(vol[:, :, x_ref], cmap='gray', aspect='auto')
    axes[2, 2].imshow(hm_combined[:, :, x_ref], cmap='hot', alpha=alpha, aspect='auto')
    axes[2, 2].set_title(f'Overlay Sagittal (x={x_ref})')
    
    # Maximum Intensity Projection (MIP)
    mip_axial = np.max(vol, axis=0)
    axes[2, 3].imshow(mip_axial, cmap='gray')
    for lm_pt in lm_voxel_valid:
        axes[2, 3].scatter(lm_pt[2], lm_pt[1], c='red', s=100, marker='x', linewidths=2)
    axes[2, 3].set_title('MIP Axial with all landmarks')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{name}_overview.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(save_dir, f'{name}_overview.png')}")


def visualize_all_landmarks(volume, landmarks, heatmaps, name, save_dir='visualizations'):
    """
    Create a detailed view showing each landmark individually.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    vol = volume.squeeze().numpy()
    lm = landmarks.numpy()
    hm = heatmaps.numpy()
    
    D, H, W = vol.shape
    
    # Denormalize
    lm_voxel = lm.copy()
    lm_voxel[:, 0] *= D
    lm_voxel[:, 1] *= H
    lm_voxel[:, 2] *= W
    
    # Filter valid landmarks
    valid_mask = ~np.all(lm == 0, axis=1)
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) == 0:
        print(f"No valid landmarks for {name}")
        return
    
    n_landmarks = len(valid_indices)
    fig, axes = plt.subplots(n_landmarks, 4, figsize=(16, 4*n_landmarks))
    
    if n_landmarks == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f'Individual Landmarks: {name}', fontsize=14)
    
    for i, lm_idx in enumerate(valid_indices):
        z, y, x = lm_voxel[lm_idx].astype(int)
        z = np.clip(z, 0, D-1)
        y = np.clip(y, 0, H-1)
        x = np.clip(x, 0, W-1)
        
        # Volume slice at landmark
        axes[i, 0].imshow(vol[z, :, :], cmap='gray')
        axes[i, 0].scatter(x, y, c='red', s=100, marker='x', linewidths=2)
        axes[i, 0].set_title(f'LM {lm_idx}: Volume (z={z})')
        
        # Individual heatmap slice
        if hm.ndim == 4:
            hm_single = hm[lm_idx]
        else:
            hm_single = hm
        
        axes[i, 1].imshow(hm_single[z, :, :], cmap='hot')
        axes[i, 1].scatter(x, y, c='cyan', s=50, marker='+', linewidths=1)
        axes[i, 1].set_title(f'LM {lm_idx}: Heatmap (z={z})')
        
        # Overlay
        axes[i, 2].imshow(vol[z, :, :], cmap='gray')
        axes[i, 2].imshow(hm_single[z, :, :], cmap='hot', alpha=0.5)
        axes[i, 2].scatter(x, y, c='cyan', s=50, marker='+', linewidths=1)
        axes[i, 2].set_title(f'LM {lm_idx}: Overlay')
        
        # Heatmap profile through landmark
        profile_y = hm_single[z, y, :]
        profile_x = hm_single[z, :, x]
        axes[i, 3].plot(profile_y, label='Along X', color='blue')
        axes[i, 3].plot(profile_x, label='Along Y', color='orange')
        axes[i, 3].axvline(x=x, color='blue', linestyle='--', alpha=0.5)
        axes[i, 3].axvline(x=y, color='orange', linestyle='--', alpha=0.5)
        axes[i, 3].set_title(f'LM {lm_idx}: Heatmap Profile')
        axes[i, 3].legend()
        axes[i, 3].set_xlabel('Position')
        axes[i, 3].set_ylabel('Intensity')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{name}_landmarks.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(save_dir, f'{name}_landmarks.png')}")


def visualize_3d_landmarks(volume, landmarks, name, save_dir='visualizations'):
    """
    Create a 3D scatter plot of landmarks in the volume.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    vol = volume.squeeze().numpy()
    lm = landmarks.numpy()
    
    D, H, W = vol.shape
    
    # Denormalize
    lm_voxel = lm.copy()
    lm_voxel[:, 0] *= D
    lm_voxel[:, 1] *= H
    lm_voxel[:, 2] *= W
    
    # Filter valid landmarks
    valid_mask = ~np.all(lm == 0, axis=1)
    lm_valid = lm_voxel[valid_mask]
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot landmarks
    if len(lm_valid) > 0:
        ax.scatter(lm_valid[:, 2], lm_valid[:, 1], lm_valid[:, 0], 
                  c='red', s=200, marker='o', label='Landmarks')
    
    # Add volume bounding box
    ax.plot([0, W], [0, 0], [0, 0], 'b-', alpha=0.3)
    ax.plot([0, 0], [0, H], [0, 0], 'b-', alpha=0.3)
    ax.plot([0, 0], [0, 0], [0, D], 'b-', alpha=0.3)
    ax.plot([W, W], [H, H], [0, D], 'b-', alpha=0.3)
    
    ax.set_xlabel('X (Width)')
    ax.set_ylabel('Y (Height)')
    ax.set_zlabel('Z (Depth)')
    ax.set_title(f'3D Landmark Positions: {name}')
    ax.legend()
    
    plt.savefig(os.path.join(save_dir, f'{name}_3d.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(save_dir, f'{name}_3d.png')}")


def visualize_slice_montage(volume, landmarks, heatmaps, name, n_slices=16, save_dir='visualizations'):
    """
    Create a montage of axial slices with landmarks and heatmaps.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    vol = volume.squeeze().numpy()
    lm = landmarks.numpy()
    hm = heatmaps.numpy()
    
    D, H, W = vol.shape
    
    # Denormalize
    lm_voxel = lm.copy()
    lm_voxel[:, 0] *= D
    lm_voxel[:, 1] *= H
    lm_voxel[:, 2] *= W
    
    valid_mask = ~np.all(lm == 0, axis=1)
    lm_valid = lm_voxel[valid_mask]
    
    # Combine heatmaps
    if hm.ndim == 4:
        hm_combined = np.max(hm, axis=0)
    else:
        hm_combined = hm
    
    # Select slice indices
    slice_indices = np.linspace(0, D-1, n_slices, dtype=int)
    
    n_cols = 4
    n_rows = (n_slices + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    fig.suptitle(f'Axial Slice Montage: {name}', fontsize=14)
    
    for i, z_idx in enumerate(slice_indices):
        row = i // n_cols
        col = i % n_cols
        
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        # Show volume
        ax.imshow(vol[z_idx, :, :], cmap='gray')
        
        # Overlay heatmap
        ax.imshow(hm_combined[z_idx, :, :], cmap='hot', alpha=0.4)
        
        # Show landmarks on this slice
        for lm_pt in lm_valid:
            if abs(lm_pt[0] - z_idx) < 2:
                ax.scatter(lm_pt[2], lm_pt[1], c='cyan', s=50, marker='+', linewidths=2)
        
        ax.set_title(f'z={z_idx}')
        ax.axis('off')
    
    # Hide empty subplots
    for i in range(len(slice_indices), n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{name}_montage.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(save_dir, f'{name}_montage.png')}")


def visualize_candidates(patch, label, world_coord, name, save_dir='visualizations'):
    """
    Visualize a candidate patch from LUNA16Candidates dataset.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    vol = patch.squeeze().numpy()
    D, H, W = vol.shape
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    label_str = "NODULE" if label == 1 else "NON-NODULE"
    fig.suptitle(f'Candidate: {name}\nLabel: {label_str}\nWorld Coord: {world_coord.numpy()}', fontsize=14)
    
    # Central slices
    z_mid, y_mid, x_mid = D//2, H//2, W//2
    
    # Row 1: Central slices
    axes[0, 0].imshow(vol[z_mid, :, :], cmap='gray')
    axes[0, 0].scatter(x_mid, y_mid, c='red', s=100, marker='+', linewidths=2)
    axes[0, 0].set_title(f'Axial (z={z_mid})')
    
    axes[0, 1].imshow(vol[:, y_mid, :], cmap='gray', aspect='auto')
    axes[0, 1].scatter(x_mid, z_mid, c='red', s=100, marker='+', linewidths=2)
    axes[0, 1].set_title(f'Coronal (y={y_mid})')
    
    axes[0, 2].imshow(vol[:, :, x_mid], cmap='gray', aspect='auto')
    axes[0, 2].scatter(y_mid, z_mid, c='red', s=100, marker='+', linewidths=2)
    axes[0, 2].set_title(f'Sagittal (x={x_mid})')
    
    # Row 2: MIP projections
    axes[1, 0].imshow(np.max(vol, axis=0), cmap='gray')
    axes[1, 0].set_title('MIP Axial')
    
    axes[1, 1].imshow(np.max(vol, axis=1), cmap='gray', aspect='auto')
    axes[1, 1].set_title('MIP Coronal')
    
    axes[1, 2].imshow(np.max(vol, axis=2), cmap='gray', aspect='auto')
    axes[1, 2].set_title('MIP Sagittal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{name}_candidate.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(save_dir, f'{name}_candidate.png')}")


def print_dataset_stats(sample):
    """Print statistics about a dataset sample."""
    print("\n" + "="*60)
    print("DATASET SAMPLE STATISTICS")
    print("="*60)
    print(f"Name: {sample['name']}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Image dtype: {sample['image'].dtype}")
    print(f"Image range: [{sample['image'].min():.4f}, {sample['image'].max():.4f}]")
    print(f"Landmarks shape: {sample['landmarks'].shape}")
    print(f"Landmarks range: [{sample['landmarks'].min():.4f}, {sample['landmarks'].max():.4f}]")
    print(f"Heatmaps shape: {sample['heatmaps'].shape}")
    print(f"Heatmaps range: [{sample['heatmaps'].min():.4f}, {sample['heatmaps'].max():.4f}]")
    print(f"Original size: {sample['original_size']}")
    print(f"Resized size: {sample['resized_size']}")
    
    # Count valid landmarks
    lm = sample['landmarks'].numpy()
    valid_mask = ~np.all(lm == 0, axis=1)
    print(f"Valid landmarks: {valid_mask.sum()} / {len(lm)}")
    
    if 'num_valid_landmarks' in sample:
        print(f"Reported valid landmarks: {sample['num_valid_landmarks']}")
    
    print("="*60 + "\n")


def main():
    """Main function to run all visualizations."""
    
    # ============================================================
    # CONFIGURATION - Update these paths for your setup
    # ============================================================
    LUNA16_PATH = '/q/AVC-AI/jkshirsagar/CSI_Project/LUNA16'  # Update this path
    SAVE_DIR = 'visualizations/luna16'
    NUM_SAMPLES = 5  # Number of samples to visualize
    
    # ============================================================
    # LUNA16 Landmark Detection Dataset
    # ============================================================
    print("\n" + "="*60)
    print("LOADING LUNA16 LANDMARK DETECTION DATASET")
    print("="*60)
    
    try:
        train_dataset = LUNA16(
            prefix=LUNA16_PATH,
            phase='train',
            size=(64, 64, 64),
            num_channels=1,
            sigma=3,
            fuse_heatmap=False,
            max_landmarks=10
        )
        
        print(f"Train dataset size: {len(train_dataset)}")
        
        # Visualize samples
        for i in range(min(NUM_SAMPLES, len(train_dataset))):
            print(f"\nProcessing sample {i+1}/{NUM_SAMPLES}...")
            sample = train_dataset[i]
            
            # Print statistics
            print_dataset_stats(sample)
            
            # Generate visualizations
            visualize_slices_with_landmarks(
                sample['image'], 
                sample['landmarks'], 
                sample['heatmaps'],
                sample['name'],
                save_dir=os.path.join(SAVE_DIR, 'train')
            )
            
            visualize_all_landmarks(
                sample['image'], 
                sample['landmarks'], 
                sample['heatmaps'],
                sample['name'],
                save_dir=os.path.join(SAVE_DIR, 'train')
            )
            
            visualize_3d_landmarks(
                sample['image'], 
                sample['landmarks'],
                sample['name'],
                save_dir=os.path.join(SAVE_DIR, 'train')
            )
            
            visualize_slice_montage(
                sample['image'], 
                sample['landmarks'], 
                sample['heatmaps'],
                sample['name'],
                save_dir=os.path.join(SAVE_DIR, 'train')
            )
            
    except Exception as e:
        print(f"Error loading LUNA16 dataset: {e}")
        import traceback
        traceback.print_exc()
    
    # ============================================================
    # LUNA16 Candidates Dataset
    # ============================================================
    print("\n" + "="*60)
    print("LOADING LUNA16 CANDIDATES DATASET")
    print("="*60)
    
    try:
        candidates_dataset = LUNA16Candidates(
            prefix=LUNA16_PATH,
            phase='train',
            size=(32, 32, 32),
            patch_size=48,
            use_v2=True
        )
        
        print(f"Candidates dataset size: {len(candidates_dataset)}")
        
        # Visualize some nodules and non-nodules
        nodule_count = 0
        non_nodule_count = 0
        
        for i in range(len(candidates_dataset)):
            if nodule_count >= 2 and non_nodule_count >= 2:
                break
                
            sample = candidates_dataset[i]
            
            if sample['class'] == 1 and nodule_count < 2:
                print(f"\nVisualizing nodule candidate {i}...")
                visualize_candidates(
                    sample['image'],
                    sample['label'],
                    sample['world_coord'],
                    f"nodule_{sample['name']}",
                    save_dir=os.path.join(SAVE_DIR, 'candidates')
                )
                nodule_count += 1
                
            elif sample['class'] == 0 and non_nodule_count < 2:
                print(f"\nVisualizing non-nodule candidate {i}...")
                visualize_candidates(
                    sample['image'],
                    sample['label'],
                    sample['world_coord'],
                    f"non_nodule_{sample['name']}",
                    save_dir=os.path.join(SAVE_DIR, 'candidates')
                )
                non_nodule_count += 1
                
    except Exception as e:
        print(f"Error loading LUNA16Candidates dataset: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print(f"Results saved to: {SAVE_DIR}")
    print("="*60)


if __name__ == '__main__':
    main()