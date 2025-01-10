import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

def reconstruct_depth_map_metric(depth_colored, global_min, global_max):
    """
    Reconstruct a single-channel depth map from a pseudo-colored depth image.

    Parameters:
    depth_colored (numpy.ndarray): Input pseudo-colored RGB depth image.
    global_min (float): Minimum depth value (corresponding to the minimum depth in the pseudo-colored image).
    global_max (float): Maximum depth value (corresponding to the maximum depth in the pseudo-colored image).

    Returns:
    numpy.ndarray: Reconstructed single-channel depth map.
    """
    # Ensure the input image is in RGB format
    depth_colored_rgb = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB) / 255.0

    # Use the Jet colormap from matplotlib
    jet_cmap = plt.get_cmap('jet')

    # Generate depth values corresponding to the Jet colormap
    depth_values = np.linspace(0, 1, 256)
    jet_colors = jet_cmap(depth_values)[:, :3]

    # Construct KDTree for indexing Jet colors
    tree = KDTree(jet_colors)

    # Reshape the input image's RGB values into a 2D array
    rgb_pixels = depth_colored_rgb.reshape(-1, 3)

    # Find the nearest color indices for each pixel
    _, closest_indices = tree.query(rgb_pixels)

    # Retrieve the corresponding depth values using the closest indices
    reconstructed_depth_normalized = depth_values[closest_indices].reshape(depth_colored.shape[0],
                                                                         depth_colored.shape[1])

    # Map the normalized depth values back to the original depth range
    reconstructed_depth = reconstructed_depth_normalized * (global_max - global_min) + global_min

    return reconstructed_depth

def reconstruct_depth_map(depth_image_path):
    """
    Reconstruct a single-channel depth map from a pseudo-color depth image.

    Parameters:
    depth_image (numpy.ndarray): Input RGB depth image (pseudo-color).

    Returns:
    numpy.ndarray: Reconstructed single-channel depth map.
    """
    # Convert RGB depth image to RGB format if necessary
    depth_image = cv2.imread(depth_image_path)  # Load your depth image
    depth_image_rgb = cv2.cvtColor(depth_image, cv2.COLOR_BGR2RGB)  # Ensure RGB format

    # Use matplotlib's jet colormap
    jet_cmap = plt.get_cmap('jet_r')

    # Generate depth values corresponding to the color mapping
    depth_values = np.linspace(0, 1, 256)  # Depth values from 0 to 1
    jet_colors = jet_cmap(depth_values)[:, :3]  # Extract RGB channels

    # Reshape RGB image to a 2D array of pixels
    rgb_pixels = depth_image_rgb.reshape(-1, 3) / 255.0

    # Build KDTree index for the jet colors
    tree = KDTree(jet_colors)

    # Find the nearest color index for each pixel
    _, closest_indices = tree.query(rgb_pixels)

    # Create the reconstructed single-channel depth map
    reconstructed_depth = depth_values[closest_indices].reshape(depth_image.shape[0], depth_image.shape[1])

    return reconstructed_depth



def reconstruct_true_depth_map(depth_image_path, colormap='jet_r'):
    """
    Reconstruct a single-channel depth map from a pseudo-color depth image, keeping the true depth values.

    Parameters:
    depth_image_path (str): Path to the input RGB depth image (pseudo-color).
    colormap (str): The colormap used for encoding the pseudo-color (default: 'jet_r').

    Returns:
    numpy.ndarray: Reconstructed single-channel depth map with the true depth values.
    """
    # Load the pseudo-color depth image
    depth_image = cv2.imread(depth_image_path)
    depth_image_rgb = cv2.cvtColor(depth_image, cv2.COLOR_BGR2RGB)

    # Load the color mapping used (assume 'jet' colormap was used, or specify your own)
    cmap = plt.get_cmap(colormap)

    # Get the depth values corresponding to the color mapping
    depth_values = np.linspace(0, 1, 256)  # Assuming depth values are in the range [0, 1]
    colormap_values = cmap(depth_values)[:, :3] * 255  # Convert to [0, 255] RGB values

    # Flatten the RGB image to a 2D array (each row is an RGB value)
    rgb_pixels = depth_image_rgb.reshape(-1, 3)

    # Build a KDTree for the colormap
    tree = KDTree(colormap_values)

    # Find the closest color in the colormap for each pixel
    _, closest_indices = tree.query(rgb_pixels)

    # Reconstruct the depth map using the depth values
    reconstructed_depth = depth_values[closest_indices].reshape(depth_image.shape[:2])

    # Scale the depth values back to their original range (if known)
    # If your original depth values are not normalized between 0 and 1, adjust this scaling.
    true_depth_map = reconstructed_depth * 255  # Adjust scaling as necessary to match your depth range

    return true_depth_map.astype(np.uint8)  # Ensure single-channel, 8-bit depth map

def knn_outlier_removal_multiscale(depth_map_batch, k_size=5, threshold=0.1, num_scales=3):
    """
    Efficient KNN outlier removal using a multi-scale strategy, supporting batch input.

    Parameters:
    - depth_map_batch: torch.Tensor, a tensor of depth maps with shape [batch_size, 1, H, W].
    - k_size: int, the size of the neighborhood window, must be an odd number.
    - threshold: float, threshold coefficient; pixels that deviate from the local median by more than 
                 threshold * local standard deviation are considered outliers.
    - num_scales: int, number of scales (i.e., Pyramid levels).

    Returns:
    - depth_filtered: torch.Tensor, depth maps after removing outliers.
    - valid_mask: torch.Tensor, a mask indicating valid depth values (valid values are 1, outliers are 0).
    """
    device = depth_map_batch.device

    # Create a multi-scale pyramid (Gaussian pyramid)
    pyramid_batch = [depth_map_batch]
    for _ in range(1, num_scales):
        # Downsample using bilinear interpolation
        depth_downsampled = F.interpolate(pyramid_batch[-1], scale_factor=0.5, mode='bilinear', align_corners=False)
        pyramid_batch.append(depth_downsampled)

    # Apply outlier detection and correction at the coarsest scale
    depth_filtered, valid_mask = knn_outlier_removal_optimized_v2(pyramid_batch[-1], k_size, threshold)

    # Upsample from the coarsest scale and correct layer by layer
    for scale in range(num_scales - 2, -1, -1):
        # Upsample to a higher resolution, ensuring sizes match the target scale
        depth_filtered = F.interpolate(depth_filtered, size=pyramid_batch[scale].shape[2:], mode='bilinear', align_corners=False)

        # Check and correct size
        if depth_filtered.shape != pyramid_batch[scale].shape:
            depth_filtered = depth_filtered[:, :, :pyramid_batch[scale].shape[2], :pyramid_batch[scale].shape[3]]

        # Apply outlier detection and correction at the current scale
        depth_filtered, valid_mask = knn_outlier_removal_optimized_v2(depth_filtered, k_size, threshold)

    return depth_filtered, valid_mask


def knn_outlier_removal_optimized_v2(depth_map_batch, k_size=5, threshold=0.1):
    """
    Apply KNN outlier removal to a batch of depth maps.

    Parameters:
    - depth_map_batch: torch.Tensor, a tensor of depth maps with shape [batch_size, 1, H, W].
    - k_size: int, the size of the neighborhood window, must be an odd number.
    - threshold: float, threshold coefficient; pixels that deviate from the local median by more than 
                 threshold * local standard deviation are considered outliers.

    Returns:
    - depth_filtered: torch.Tensor, depth maps after removing outliers.
    - valid_mask: torch.Tensor, a mask indicating valid depth values (valid values are 1, outliers are 0).
    """
    device = depth_map_batch.device
    pad = k_size // 2

    # Set pixels with values of 0 and 1 to NaN, indicating invalid
    depth_map = depth_map_batch.clone()
    invalid_mask = (depth_map == 0) | (depth_map == 1)
    depth_map[invalid_mask] = float('nan')

    # Create mean and median filters
    kernel = torch.ones((1, 1, k_size, k_size), device=device)

    # Calculate the count of valid pixels (considering NaN)
    depth_padded = F.pad(depth_map, (pad, pad, pad, pad), mode='reflect')
    valid_mask_padded = (~torch.isnan(depth_padded)).float()
    valid_count = F.conv2d(valid_mask_padded, kernel, padding=0)
    valid_count = valid_count - (~torch.isnan(depth_map)).float()  # Exclude center pixel

    # Calculate local sum (ignoring NaN)
    depth_padded_zero = torch.nan_to_num(depth_padded, nan=0.0)
    depth_sum = F.conv2d(depth_padded_zero, kernel, padding=0)
    center_pixel = torch.nan_to_num(depth_map, nan=0.0) * (k_size * k_size - 1)
    local_sum = depth_sum - center_pixel
    local_mean = local_sum / (valid_count + 1e-6)

    # Calculate local median
    depth_patches = depth_padded.unfold(2, k_size, 1).unfold(3, k_size, 1)
    depth_patches = depth_patches.contiguous().view(depth_patches.size(0), depth_patches.size(1), depth_patches.size(2), depth_patches.size(3), -1)

    local_median = depth_patches.median(dim=-1)[0]
    local_median = local_median.view_as(depth_map)

    # Calculate local variance and standard deviation
    local_var = torch.var(depth_patches, dim=-1, unbiased=False).view_as(depth_map)
    local_std = torch.sqrt(local_var + 1e-6)

    # Detect outliers using local median for judgement
    abs_diff = torch.abs(depth_map - local_median)
    outlier_mask = (abs_diff > threshold * local_std) & (~torch.isnan(depth_map))
    outlier_mask = outlier_mask.float()

    # Replace outliers with local median
    depth_filtered = depth_map.clone()
    depth_filtered[outlier_mask == 1] = local_median[outlier_mask == 1]

    # Set previous invalid values (0 and 1) back to NaN
    depth_filtered[invalid_mask] = float('nan')

    # Generate valid value mask
    valid_mask = (~torch.isnan(depth_filtered)).float()

    return depth_filtered, valid_mask



