import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import requests
import torch
from GPUtil import showUtilization as gpu_usage
from prettytable import PrettyTable

## -----------------------------------------------------------------------------------------------------------------##
##                                          CLEAN GPU MEMORY USAGE AND DATASETS                                     ##
## -----------------------------------------------------------------------------------------------------------------##

def free_gpu_cache():
    print("Initial GPU Usage")
    gpu_usage() 
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    print("GPU Usage after emptying the cache")
    gpu_usage()

# Compute the number of trainable parameters in a model
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    #print(table)
    print(f"Total Trainable Params: {total_params}")
    return table, total_params

## -----------------------------------------------------------------------------------------------------------------##
##                            HEATMAPS GENERATION FROM LANDMARKS POINTS                                             ##
## -----------------------------------------------------------------------------------------------------------------##

def fuse_heatmaps(heatmaps):
    fused_heatmap = np.sum(heatmaps, axis=0)

    # Threshold the heatmap so that values below a certain threshold are set to 0
    binary_fused_heatmap = np.where(fused_heatmap < 0.5, 0, 1)

    assert is_binary_image(binary_fused_heatmap), "Image is not binary"

    return binary_fused_heatmap

def fuse_heatmaps(heatmaps):
    # Use np.maximum.reduce to get the maximum value from each stack of pixels across the heatmaps
    fused_heatmap = np.maximum.reduce(heatmaps)
    return fused_heatmap

def scale_points(points: list, img_size: tuple, orig_size: tuple = None, offset=0):    
    # Scale coordinates according to image size
    if orig_size:
        # Points * Ratio -> Downscaling
        scaled_points = [tuple([round(p*isize/osize)+offset for p, isize, osize in zip(point, img_size, orig_size)]) for point in points]
    else:
        # Points * current Size -> Upscaling (when i use "extract_landmarks" function the points are with size (1,1))
        scaled_points = [tuple([round(r*sz)+offset for sz, r in zip(point, img_size)]) for point in points]

    return scaled_points


def points_to_heatmap(points: list, img_size: tuple, orig_size: tuple = None, sigma=2, fuse=False, offset=0):

    # Scale coordinates according to image size
    if orig_size:
        scaled_points = scale_points(points, img_size, orig_size, offset=offset)
    else:
        scaled_points = scale_points(points, img_size, offset=offset)

    # Generate heatmaps with the scaled points
    heatmaps = [generate_heatmap_from_points(point, img_size, sigma) for point in scaled_points]

    if fuse:
        heatmaps = fuse_heatmaps(heatmaps)

    return np.array(heatmaps)


def generate_heatmap_from_points(point, img_size, sigma):
    # Create a meshgrid of x,y coordinates for the image size
    x, y = np.meshgrid(np.arange(img_size[0]), np.arange(img_size[1]))
    # Calculate the heatmap using a Gaussian function centered at the input point
    heatmap = np.exp(-((x - point[0]) ** 2 + (y - point[1]) ** 2) / (2 * sigma ** 2))
    # Threshold the heatmap so that values below a certain threshold are set to 0
    binary_heatmap = np.where(heatmap < 0.5*heatmap.max(), 0, 1)

    assert is_binary_image(binary_heatmap), "Image is not binary"

    return binary_heatmap

def is_binary_image(image):
    binary_check = np.logical_or(image == 0, image == 1)
    return binary_check.all()
## -----------------------------------------------------------------------------------------------------------------##
##                                   LANDMARKS EXTRACTION FROM HEATMAPS                                             ##
## -----------------------------------------------------------------------------------------------------------------##

def extract_landmarks_3d(heatmaps: np.ndarray, num_landmarks: int = None, threshold: float = None):
    """
    Extract 3D landmarks from heatmaps using weighted centroid in voxel space.
    
    Args:
        heatmaps: (N, D, H, W) or (D, H, W). If fused, pass (D, H, W).
        num_landmarks: Number of landmarks to return for (N, D, H, W). If None, uses N.
        threshold: Optional fractional threshold of max to zero-out low values (e.g., 0.1).
    
    Returns:
        landmarks: (K, 3) array of normalized coordinates in [0,1], ordered (z, y, x).
                   If a channel has no signal, returns [-1, -1, -1] for that entry.
                   K = 1 for fused input; K = min(N, num_landmarks) for multi-channel input.
    """
    if heatmaps.ndim == 3:
        # Fused heatmap case -> single landmark
        hm = heatmaps.astype(np.float32)
        D, H, W = hm.shape
        if threshold is not None and hm.max() > 0:
            hm = np.where(hm >= threshold * hm.max(), hm, 0.0)

        total = hm.sum()
        if total <= 0:
            return np.array([[-1.0, -1.0, -1.0]], dtype=np.float32)

        # Weighted centroid in voxel space (z, y, x)
        z_idx, y_idx, x_idx = np.meshgrid(
            np.arange(D), np.arange(H), np.arange(W), indexing='ij'
        )
        cz = (hm * z_idx).sum() / total
        cy = (hm * y_idx).sum() / total
        cx = (hm * x_idx).sum() / total

        # Normalize to [0,1]
        return np.array([[cz / D, cy / H, cx / W]], dtype=np.float32)

    assert heatmaps.ndim == 4, f"extract_landmarks_3d expects (N,D,H,W) or (D,H,W), got {heatmaps.shape}"
    N, D, H, W = heatmaps.shape
    K = N if num_landmarks is None else min(N, num_landmarks)

    landmarks = np.full((K, 3), -1.0, dtype=np.float32)

    # Precompute coordinate grids
    z_idx, y_idx, x_idx = np.meshgrid(
        np.arange(D), np.arange(H), np.arange(W), indexing='ij'
    )

    for i in range(K):
        hm = heatmaps[i].astype(np.float32)

        if threshold is not None and hm.max() > 0:
            hm = np.where(hm >= threshold * hm.max(), hm, 0.0)

        total = hm.sum()
        if total <= 0:
            # No signal -> invalid landmark
            landmarks[i] = [-1.0, -1.0, -1.0]
            continue

        cz = (hm * z_idx).sum() / total
        cy = (hm * y_idx).sum() / total
        cx = (hm * x_idx).sum() / total

        landmarks[i] = [cz / D, cy / H, cx / W]

    return landmarks

def extract_landmarks(heatmaps, num_landmarks):
    # If heatmaps are 4D (N, D, H, W) -> use 3D extractor
    if heatmaps.ndim == 4:
        return extract_landmarks_3d(heatmaps, num_landmarks)

    # If heatmaps are 3D but last dim is channels, reshape/transpose so that
    # we have (N, H, W)
    if heatmaps.ndim == 3:
        # Assume shape (N, H, W); if not, you may need to adjust this logic
        pass

    # Now expect shape (N, H, W)
    assert heatmaps.ndim == 3, f"extract_landmarks expects heatmaps with shape (N, H, W), got {heatmaps.shape}"

    landmarks = np.zeros((num_landmarks, 2), dtype=np.float64)
    heatmap_height, heatmap_width = heatmaps.shape[1], heatmaps.shape[2]

    for i in range(num_landmarks):
        heatmap_channel = heatmaps[i]  # (H, W)

        # binarize the heatmap channel using a threshold
        binary_img = np.where(
            heatmap_channel < 0.5 * heatmap_channel.max(), 0, 255
        ).astype(np.uint8)

        # Ensure 2D single-channel image for OpenCV
        if binary_img.ndim == 3:
            # If shape is (H, W, 1), squeeze; if (1, H, W), take first slice
            if binary_img.shape[-1] == 1:
                binary_img = binary_img[..., 0]
            elif binary_img.shape[0] == 1:
                binary_img = binary_img[0]
            else:
                raise ValueError(f"binary_img has invalid shape for findContours: {binary_img.shape}")

        contours, _ = cv2.findContours(
            binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(max_contour)

            if M['m00'] != 0:
                centroid_x = M['m10'] / M['m00']  # along width (x/cols)
                centroid_y = M['m01'] / M['m00']  # along height (y/rows)
                landmarks[i, :] = [
                    centroid_x / heatmap_width,   # divide by width
                    centroid_y / heatmap_height,  # divide by height
                ]
    return landmarks


## -----------------------------------------------------------------------------------------------------------------##
##                                                  PLOT LOSS CURVES                                                ##
## -----------------------------------------------------------------------------------------------------------------##

def plot_loss_curves(results_path: str, save_dir: str = None):
    # Load the results dictionary
    results = torch.load(results_path)['results']
    # Get the loss values of the results dictionary (training and validation)
    train_loss = results['train_loss']
    val_loss = results['val_loss']

    # Figure out how many epochs there were
    epochs = range(1, len(results['train_loss']) + 1)

    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss, label='train_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, f"loss_epochs{epochs}"))

    plt.show()

# -----------------------------------------------------------------------------------------------------------------##
# 
# -----------------------------------------------------------------------------------------------------------------##

# Function to generate the save model path for both custom model and segmentation model.
def generate_save_model_path(PREFIX, model_name, dataset_name, sigma, size, pretrained=None, backbone=None):
    if pretrained is not None:
        pretrained_dir = pretrained
    else:
        pretrained_dir = "no_pretrain"

    if backbone is not None:
        backbone_dir = backbone
    else:
        backbone_dir = "no_backbone"

    save_model_path = f'{PREFIX}/results/models/{model_name}/{pretrained_dir}/{backbone_dir}/{dataset_name}/sigma{sigma}_size{str(size).replace(", ", "x")}'
    return save_model_path

# Generate path if it does not exist
def generate_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

# Save the: original image with the heatmaps overlayed, single heatmaps and fused heatmap
def save_heatmaps(batch_images, batch_heatmaps, images_name, save_dir):
    for i, sample in enumerate(batch_heatmaps):
        
        original_image = batch_images[i].permute(1, 2, 0).cpu().numpy()
        
        plt.figure(figsize=(10, 10))
        plt.imshow(original_image)
        for j, heatmap in enumerate(sample):
            plt.imshow(heatmap, cmap='viridis', alpha=0.25)
            
            # Save the single heatmaps
            plt.imsave(f"{save_dir}/{images_name[i]}_heatmap_{j}.png", heatmap, cmap='viridis')

        # Save the original image with the heatmaps overlayed
        plt.savefig(f"{save_dir}/{images_name[i]}_overlayed_heatmaps.png")
        plt.close()
        
        # Save the fused heatmap
        fused_heatmap = fuse_heatmaps(sample)
        plt.imsave(f"{save_dir}/{images_name[i]}_fused_heatmap.png", fused_heatmap, cmap='viridis')
        


# -----------------------------------------------------------------------------------------------------------------##

# Load env variables from a .env file
def load_env_variables(env_file):
    with open(env_file, 'r') as f:
        for line in f:
            key, value = line.strip().split('=')
            # trim whitespace
            key = key.strip()
            value = value.strip()
            # remove quotes if present
            if value[0] == value[-1] and value.startswith(("'", '"')):
                value = value[1:-1]

            os.environ[key] = value

def send_telegram_message(text, env_file='~/.env'):
    env_file_path = os.path.expanduser(env_file)

    if os.path.exists(env_file_path):
        load_env_variables(env_file_path)
    else:
        print(f"Error: {env_file_path} does not exist")
        return
    
    token = os.getenv('TELEGRAM_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    url_req = "https://api.telegram.org/bot" + token + "/sendMessage"
    payload = {
        'chat_id': chat_id,
        'text': text,
        'parse_mode': 'HTML' #or HTML or MarkdownV2
    }
    
    try:
        response = requests.get(url_req, params=payload)
    except Exception as e:
        print(f"Error sending telegram message: {e}")

    return response
        
        
## -----------------------------------------------------------------------------------------------------------------##
##                            3D HEATMAPS GENERATION FROM LANDMARKS POINTS                                          ##
## -----------------------------------------------------------------------------------------------------------------##

def fuse_heatmaps_3d(heatmaps):
    """Fuse multiple 3D heatmaps into one by taking maximum value at each voxel."""
    fused_heatmap = np.maximum.reduce(heatmaps)
    return fused_heatmap

def scale_points_3d(points: list, vol_size: tuple, orig_size: tuple = None, offset=0):    
    """
    Scale 3D coordinates according to volume size.
    Note: points are ordered (z, y, x) to match vol_size (D, H, W).
    Args:
        points: List/array of 3D points (z, y, x) normalized to [0,1] or absolute
        vol_size: Target volume size (D, H, W)
        orig_size: Original volume size (optional)
        offset: Offset to add to scaled points
    """
    # Scale coordinates according to volume size
    if orig_size:
        # Points * Ratio -> Downscaling
        scaled_points = [tuple([round(p*vsize/osize)+offset for p, vsize, osize in zip(point, vol_size, orig_size)]) for point in points]
    else:
        # Points * current Size -> Upscaling (when points are normalized to [0, 1])
        scaled_points = [tuple([round(r*sz)+offset for sz, r in zip(vol_size, point)]) for point in points]

    return scaled_points


def points_to_heatmap_3d(points: np.ndarray, vol_size: tuple, sigma=2, fuse=False, debug=False):
    """
    Generate 3D heatmaps from point coordinates.
    
    Args:
        points: (N, 3) array of coordinates in VOXEL space (z, y, x) - already scaled to vol_size
        vol_size: (D, H, W) - size of output heatmap volume
        sigma: Gaussian sigma
        fuse: Whether to fuse all heatmaps
        debug: Whether to print debug information
    
    Returns:
        heatmaps: (N, D, H, W) array if fuse=False, (D, H, W) if fuse=True
    """
    if len(points) == 0:
        if fuse:
            return np.zeros(vol_size, dtype=np.float32)
        else:
            return np.zeros((0, *vol_size), dtype=np.float32)
    
    # Filter out invalid points (those with -1 values from padding)
    valid_mask = np.all(points >= 0, axis=1)
    valid_points = points[valid_mask]
    
    if debug:
        print(f"[points_to_heatmap_3d] Total points: {len(points)}, Valid: {len(valid_points)}")
        print(f"[points_to_heatmap_3d] Vol size: {vol_size}")
        for i, pt in enumerate(valid_points):
            print(f"  Point {i}: z={pt[0]:.2f}, y={pt[1]:.2f}, x={pt[2]:.2f}")
    
    if len(valid_points) == 0:
        if fuse:
            return np.zeros(vol_size, dtype=np.float32)
        else:
            return np.zeros((len(points), *vol_size), dtype=np.float32)
    
    # Generate heatmaps only for valid points
    heatmaps = []
    for i, point in enumerate(valid_points):
        hm = generate_heatmap_from_points_3d(point, vol_size, sigma)
        if debug:
            print(f"  Heatmap {i}: max={hm.max():.4f}, sum={hm.sum():.4f}")
        heatmaps.append(hm)
    
    heatmaps = np.array(heatmaps)
    
    if fuse:
        return fuse_heatmaps_3d(heatmaps)
    
    return heatmaps


def generate_heatmap_from_points_3d(point, vol_size, sigma):
    """
    Generate continuous 3D Gaussian heatmap.
    
    Args:
        point: (z, y, x) coordinates in voxel space (already scaled to vol_size)
        vol_size: (D, H, W) - volume dimensions
        sigma: Standard deviation for Gaussian
    
    Returns:
        heatmap: (D, H, W) array with Gaussian centered at point
    """
    # Create coordinate grids
    # meshgrid with indexing='ij' gives us (D, H, W) shaped arrays
    # where z varies along axis 0, y along axis 1, x along axis 2
    z, y, x = np.meshgrid(
        np.arange(vol_size[0]),  # D dimension (z)
        np.arange(vol_size[1]),  # H dimension (y)  
        np.arange(vol_size[2]),  # W dimension (x)
        indexing='ij'
    )
    
    # Point is (z, y, x) in voxel coordinates
    pz, py, px = point[0], point[1], point[2]
    
    # Compute squared distances
    dist_sq = (z - pz) ** 2 + (y - py) ** 2 + (x - px) ** 2
    
    # Generate Gaussian heatmap
    heatmap = np.exp(-dist_sq / (2 * sigma ** 2))
    
    # Normalize to [0, 1]
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    return heatmap.astype(np.float32)

## -----------------------------------------------------------------------------------------------------------------##
##                                   DEBUG AND VISUALIZATION UTILITIES                                              ##
## -----------------------------------------------------------------------------------------------------------------##

def debug_landmark_heatmap_3d(sample, slice_idx=None):
    """
    Debug utility to visualize 3D landmarks and heatmaps.
    
    Args:
        sample: Dictionary from dataset __getitem__ containing:
            - 'image': (C, D, H, W) tensor
            - 'heatmaps': (N, D, H, W) tensor
            - 'landmarks': (N, 3) tensor (normalized coordinates)
            - 'num_valid_landmarks': int
            - 'resized_size': (D, H, W) tensor
        slice_idx: Optional specific slice indices (z, y, x) to visualize.
                   If None, uses center of volume.
    """
    import matplotlib.pyplot as plt
    
    image = sample['image'].numpy() if torch.is_tensor(sample['image']) else sample['image']
    heatmaps = sample['heatmaps'].numpy() if torch.is_tensor(sample['heatmaps']) else sample['heatmaps']
    landmarks = sample['landmarks'].numpy() if torch.is_tensor(sample['landmarks']) else sample['landmarks']
    num_valid = sample.get('num_valid_landmarks', len(landmarks))
    vol_size = sample['resized_size'].numpy() if torch.is_tensor(sample['resized_size']) else sample['resized_size']
    
    # Handle channel dimension
    if image.ndim == 4:
        image = image[0]  # Take first channel
    
    D, H, W = image.shape
    
    print(f"=== Debug Info ===")
    print(f"Image shape: {image.shape}, range: [{image.min():.3f}, {image.max():.3f}]")
    print(f"Heatmaps shape: {heatmaps.shape}, range: [{heatmaps.min():.4f}, {heatmaps.max():.4f}]")
    print(f"Num valid landmarks: {num_valid}")
    print(f"Volume size: {vol_size}")
    
    # Print landmark info
    for i in range(num_valid):
        lm = landmarks[i]
        # Convert normalized to voxel coordinates
        voxel_coords = lm * vol_size
        print(f"Landmark {i}: normalized=({lm[0]:.3f}, {lm[1]:.3f}, {lm[2]:.3f}) -> "
              f"voxel=(z={voxel_coords[0]:.1f}, y={voxel_coords[1]:.1f}, x={voxel_coords[2]:.1f})")
        
        # Check heatmap at this location
        if heatmaps.ndim == 4:  # (N, D, H, W)
            z_idx = int(np.clip(voxel_coords[0], 0, D-1))
            y_idx = int(np.clip(voxel_coords[1], 0, H-1))
            x_idx = int(np.clip(voxel_coords[2], 0, W-1))
            hm_val = heatmaps[i, z_idx, y_idx, x_idx] if i < heatmaps.shape[0] else 0
            print(f"  Heatmap {i} value at landmark: {hm_val:.4f}, max: {heatmaps[i].max():.4f}")
    
    # Determine slice indices
    if slice_idx is None:
        # Use center or first valid landmark location
        if num_valid > 0:
            center_landmark = landmarks[0] * vol_size
            slice_z = int(np.clip(center_landmark[0], 0, D-1))
            slice_y = int(np.clip(center_landmark[1], 0, H-1))
            slice_x = int(np.clip(center_landmark[2], 0, W-1))
        else:
            slice_z, slice_y, slice_x = D//2, H//2, W//2
    else:
        slice_z, slice_y, slice_x = slice_idx
    
    print(f"\nVisualizing slices: z={slice_z}, y={slice_y}, x={slice_x}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Top row: Image slices
    axes[0, 0].imshow(image[slice_z, :, :], cmap='gray')
    axes[0, 0].set_title(f'Axial (z={slice_z})')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    
    axes[0, 1].imshow(image[:, slice_y, :], cmap='gray')
    axes[0, 1].set_title(f'Coronal (y={slice_y})')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Z')
    
    axes[0, 2].imshow(image[:, :, slice_x], cmap='gray')
    axes[0, 2].set_title(f'Sagittal (x={slice_x})')
    axes[0, 2].set_xlabel('Y')
    axes[0, 2].set_ylabel('Z')
    
    # Bottom row: Fused heatmaps overlaid
    if heatmaps.ndim == 4:
        fused = np.maximum.reduce(heatmaps[:num_valid]) if num_valid > 0 else np.zeros_like(image)
    else:
        fused = heatmaps
    
    axes[1, 0].imshow(image[slice_z, :, :], cmap='gray')
    axes[1, 0].imshow(fused[slice_z, :, :], cmap='hot', alpha=0.5)
    axes[1, 0].set_title(f'Axial + Heatmap (max={fused[slice_z].max():.3f})')
    
    axes[1, 1].imshow(image[:, slice_y, :], cmap='gray')
    axes[1, 1].imshow(fused[:, slice_y, :], cmap='hot', alpha=0.5)
    axes[1, 1].set_title(f'Coronal + Heatmap (max={fused[:, slice_y, :].max():.3f})')
    
    axes[1, 2].imshow(image[:, :, slice_x], cmap='gray')
    axes[1, 2].imshow(fused[:, :, slice_x], cmap='hot', alpha=0.5)
    axes[1, 2].set_title(f'Sagittal + Heatmap (max={fused[:, :, slice_x].max():.3f})')
    
    # Mark landmark positions
    for i in range(num_valid):
        lm = landmarks[i] * vol_size
        z, y, x = lm[0], lm[1], lm[2]
        
        # Axial view: mark if landmark is near this slice
        if abs(z - slice_z) < 3:
            axes[0, 0].plot(x, y, 'g+', markersize=10, markeredgewidth=2)
            axes[1, 0].plot(x, y, 'g+', markersize=10, markeredgewidth=2)
        
        # Coronal view
        if abs(y - slice_y) < 3:
            axes[0, 1].plot(x, z, 'g+', markersize=10, markeredgewidth=2)
            axes[1, 1].plot(x, z, 'g+', markersize=10, markeredgewidth=2)
        
        # Sagittal view
        if abs(x - slice_x) < 3:
            axes[0, 2].plot(y, z, 'g+', markersize=10, markeredgewidth=2)
            axes[1, 2].plot(y, z, 'g+', markersize=10, markeredgewidth=2)
    
    plt.tight_layout()
    plt.savefig('debug_heatmap_3d.png', dpi=150)
    plt.show()
    
    return fig
