import numpy as np
import scipy.spatial.distance as dist
import utilities
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

## -----------------------------------------------------------------------------------------------------------------##
##                                                  Mean Squared Error                                             ##
## -----------------------------------------------------------------------------------------------------------------##

def compute_mse(gt_keypoints, pred_keypoints):
    assert gt_keypoints.shape == pred_keypoints.shape, "The ground truth list has not the same shape of the predicted list"

    # Compute squared differences
    squared_diff = np.square(gt_keypoints - pred_keypoints)

    # Compute mean
    mse = np.mean(squared_diff)

    return mse

## -----------------------------------------------------------------------------------------------------------------##
##                                                  mAp with OKS for heatmaps                                       ##
## -----------------------------------------------------------------------------------------------------------------##
def compute_oks_heatmaps(ground_truth, prediction, sigma):
    """
    Compute OKS between two (possibly multi‑dimensional) heatmaps by flattening them.
    ground_truth, prediction: arrays with the same shape, e.g. (H, W) or (D, H, W).
    """
    gt_flat = ground_truth.reshape(1, -1)   # (1, N)
    pred_flat = prediction.reshape(1, -1)   # (1, N)
    distance = dist.cdist(gt_flat, pred_flat, 'euclidean')  # (1, 1)
    scale = 1
    oks = np.exp(-1 * (distance ** 2) / (2 * (sigma**2) * (scale ** 2)))
    return oks

def compute_map_heatmaps(ground_truth_heatmaps, predicted_heatmaps, sigma=0.1, thresholds=np.arange(0.5, 1.0, 0.05)):
    """
    Compute a toy mAP-like score between two heatmaps (2D or 3D) by:
      1) computing a single OKS value between flattened maps
      2) thresholding that scalar at multiple levels.
    This keeps the interface but avoids shape errors.
    """
    aps = []

    assert ground_truth_heatmaps.shape == predicted_heatmaps.shape, "Heatmaps should have the same shape"

    oks = compute_oks_heatmaps(ground_truth_heatmaps, predicted_heatmaps, sigma)  # shape (1,1)
    oks_scalar = float(oks[0, 0])

    for threshold in thresholds:
        tp = 1 if oks_scalar >= threshold else 0
        fp = 1 - tp
        precision = tp / (tp + fp)  # either 1 or 0
        aps.append(precision)

    map_value = np.mean(aps)
    return map_value


## -----------------------------------------------------------------------------------------------------------------##
##                                                  mAp with OKS for keypoints                                       ##
## -----------------------------------------------------------------------------------------------------------------##
def compute_oks_keypoints(ground_truth, prediction, sigma, axis_scale=None):
    """
    axis_scale: per-axis physical scaling (e.g. voxel spacing). If provided, distance computed after scaling.
    """
    diff = ground_truth - prediction
    if axis_scale is not None:
        diff = diff * axis_scale  # broadcast (N,3) * (3,)
    distance = np.sqrt(np.sum(diff**2, axis=1))
    scale = 1
    oks = np.exp(- (distance ** 2) / (2 * (sigma ** 2) * (scale ** 2)))
    return oks


def compute_map_keypoints(ground_truth_keypoints, predicted_keypoints, sigma=0.1, thresholds=np.arange(0.5, 1.0, 0.05)):
    aps = []

    # Calculate OKS value
    oks = compute_oks_keypoints(ground_truth_keypoints, predicted_keypoints, sigma)
    for threshold in thresholds:
        # Calculate precision
        tp = np.sum(oks >= threshold)
        fp = np.sum(oks < threshold)
        precision = tp / (tp + fp)
        aps.append(precision)
    # Calculate the mean average precision
    map_value = np.mean(aps)
    return map_value



## -----------------------------------------------------------------------------------------------------------------##
##                                                  Intersection Over Union                                       ##
## -----------------------------------------------------------------------------------------------------------------##

def compute_iou_heatmaps(heatmap1, heatmap2):

    assert heatmap1.shape == heatmap2.shape, "Heatmaps should have the same shape"

    overlap = np.logical_and(heatmap1, heatmap2)
    union = np.logical_or(heatmap1, heatmap2)
    overlap_area = np.sum(overlap)
    union_area = np.sum(union)
    IoU = overlap_area / union_area
    return IoU



## -----------------------------------------------------------------------------------------------------------------##
##                                                 Aux functions                                       ##
## -----------------------------------------------------------------------------------------------------------------##

from collections.abc import Iterable

def radial(pt1, pt2, factor=1):
    if  not isinstance(factor,Iterable):
        factor = [factor]*len(pt1)
    return sum(((i-j)*s)**2 for i, j,s  in zip(pt1, pt2, factor))**0.5

def cal_all_distance(points, gt_points, factor=1):
    '''
    points: [(x,y,z...)]
    gt_points: [(x,y,z...)]
    return : [d1,d2, ...]
    '''
    n1 = len(points)
    n2 = len(gt_points)
    if n1 == 0:
        print("[Warning]: Empty input for calculating mean and std")
        return 0, 0
    if n1 != n2:
        raise Exception("Error: lengthes dismatch, {}<>{}".format(n1, n2))
    return [radial(p, q, factor) for p, q in zip(points, gt_points)]


## -----------------------------------------------------------------------------------------------------------------##
##                                                  Mean Radial Error (MRE)                                       ##
## -----------------------------------------------------------------------------------------------------------------##

"""
MRE (Mean Radial Error): 
This measures the average euclidean distance between predicted landmarks and ground truth landmarks. 
It is calculated by taking the mean of the list of distances (cal_all_distance).
"""

def compute_mre(distance_list):
    return np.mean(distance_list)

## -----------------------------------------------------------------------------------------------------------------##
##                                                  Successful Detection Rate (SDR)                                       ##
## -----------------------------------------------------------------------------------------------------------------##
"""
SDR (Successful Detection Rate): 
This measures the percentage of predicted landmarks that are within a threshold distance of the ground truth. 
It is calculated by get_sdr which counts the number of distances below each threshold and divides by the total number of landmarks.
"""

def compute_sdr(distance_list, threshold=[10, 15, 20, 25, 30]):
    """
    Compute Successful Detection Rate (SDR) in pixel for a given list of distances and thresholds.
    The SDR is the proportion of predicted points that fall within a certain distance threshold from the ground truth points.
    """
    sdr = {}
    n = len(distance_list)

    for th in threshold:
        sdr[th] = sum(d <= th for d in distance_list) / n
    return sdr

## -----------------------------------------------------------------------------------------------------------------##
##                                                  COMPUTE BATCH METRICS                                       ##
## -----------------------------------------------------------------------------------------------------------------##


def compute_batch_metrics(gt_batch_keypoints, gt_batch_heatmaps, pred_batch, image_size, num_landmarks, useHeatmaps, sigma, spacing_batch=None, fused=False, num_valid_batch=None):
    """
    Compute batch metrics for 2D/3D.
    
    Args:
        num_valid_batch: torch.Tensor [B] containing number of valid landmarks per sample
                        (required for datasets with variable landmark counts like LUNA16)
    """
    # Ensure numpy
    if hasattr(pred_batch, "detach"):
        pred_batch = pred_batch.detach().numpy()

    batch_size = pred_batch.shape[0]
    mse_list, map_list1, map_list2, iou_list, distance_list = [], [], [], [], []

    for i in range(batch_size):
        single_gt_keypoints = gt_batch_keypoints[i].numpy()  # normalized
        single_gt_heatmaps = gt_batch_heatmaps[i].numpy()
        single_pred_maps = pred_batch[i]  # numpy
        grid_size = tuple(image_size[i].int().tolist())
        
        # Get number of valid landmarks for this sample
        if num_valid_batch is not None:
            num_valid = int(num_valid_batch[i].item())
        else:
            num_valid = num_landmarks  # Assume all are valid

        is_3d = (single_pred_maps.ndim == 4)  # (C,D,H,W) for 3D

        if is_3d:
            # Extract predicted keypoints from model heatmaps
            if useHeatmaps:
                pred_keypoints_all = utilities.extract_landmarks_3d(single_pred_maps, num_landmarks)
            else:
                pred_keypoints_all = single_pred_maps  # coordinate regression path

            # **FIX: Only use valid landmarks for metrics**
            gt_keypoints = single_gt_keypoints[:num_valid]  # Only valid GT
            pred_keypoints = pred_keypoints_all[:num_valid]  # Only valid predictions

            # Build fused heatmaps from points on the evaluation grid
            gt_heat_fused = utilities.points_to_heatmap_3d(gt_keypoints, vol_size=grid_size, sigma=sigma, fuse=True)
            pred_heat_fused = utilities.points_to_heatmap_3d(pred_keypoints, vol_size=grid_size, sigma=sigma, fuse=True)

            # Scale to voxel indices
            gt_vox = np.array(utilities.scale_points_3d(gt_keypoints, grid_size))
            pred_vox = np.array(utilities.scale_points_3d(pred_keypoints, grid_size))

            # If spacing (dz,dy,dx) is provided, convert to mm
            if spacing_batch is not None:
                sp = spacing_batch[i].numpy()  # (dz, dy, dx)
                gt_phys = gt_vox * sp
                pred_phys = pred_vox * sp
                use_points_for_dist = (pred_phys, gt_phys)
            else:
                use_points_for_dist = (pred_vox, gt_vox)

            # Distances and MSE (voxel or mm consistent)
            cur_distance_list = cal_all_distance(use_points_for_dist[0], use_points_for_dist[1], factor=1)
            distance_list += cur_distance_list
            mse_list.append(compute_mse(use_points_for_dist[1], use_points_for_dist[0]))

            # Keypoint mAP only when per-landmark maps are meaningful
            if not fused:
                map_list2.append(compute_map_keypoints(gt_vox, pred_vox))
            # Heatmap mAP & IoU on fused maps
            map_list1.append(compute_map_heatmaps(gt_heat_fused, pred_heat_fused))
            iou_list.append(compute_iou_heatmaps(gt_heat_fused, pred_heat_fused))

        else:
            # 2D path (unchanged)
            if useHeatmaps:
                pred_keypoints = utilities.extract_landmarks(single_pred_maps, num_landmarks)
            else:
                pred_keypoints = single_pred_maps

            gt_keypoints_extracted = utilities.extract_landmarks(single_gt_heatmaps, num_landmarks)
            gt_heat_fused = utilities.points_to_heatmap(gt_keypoints_extracted, img_size=grid_size, sigma=sigma, fuse=True)
            pred_heat_fused = utilities.points_to_heatmap(pred_keypoints, img_size=grid_size, sigma=sigma, fuse=True)

            gt_pix = np.array(utilities.scale_points(gt_keypoints_extracted, grid_size))
            pred_pix = np.array(utilities.scale_points(pred_keypoints, grid_size))

            cur_distance_list = cal_all_distance(pred_pix, gt_pix, factor=1)
            distance_list += cur_distance_list

            mse_list.append(compute_mse(gt_pix, pred_pix))
            map_list2.append(compute_map_keypoints(gt_pix, pred_pix))
            map_list1.append(compute_map_heatmaps(gt_heat_fused, pred_heat_fused))
            iou_list.append(compute_iou_heatmaps(gt_heat_fused, pred_heat_fused))

    return mse_list, map_list1, map_list2, iou_list, distance_list
