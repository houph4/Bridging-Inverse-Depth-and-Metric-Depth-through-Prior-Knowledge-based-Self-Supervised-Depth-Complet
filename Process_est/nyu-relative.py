import os
import cv2
import torch
import h5py
import numpy as np
from da2.depth_anything_v2.dpt import DepthAnythingV2
from tqdm import tqdm

def generate_all_h5_paths(root_folder):
    """
    Traverse the main folder and all subfolders to generate paths for all H5 files.
    """
    h5_paths = []
    for dirpath, _, filenames in os.walk(root_folder):
        for file in filenames:
            if file.endswith('.h5'):
                h5_paths.append(os.path.join(dirpath, file))
    return sorted(h5_paths)

def process_depth_image(model, rgb_image):
    """
    Perform depth estimation on a single RGB image.
    """
    rgb_image = np.transpose(rgb_image, (1, 2, 0))  # Convert to (H, W, C)
    raw_img_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)  # Convert to BGR format

    with torch.no_grad():
        depth_predicted = model.infer_image(raw_img_rgb).squeeze()
    return depth_predicted

def write_est_to_h5(h5_file_path, depth_predicted):
    """
    Write the estimated depth map into the H5 file under the key 'est'.
    """
    with h5py.File(h5_file_path, 'r+') as h5_file:
        if 'est_r' in h5_file:
            del h5_file['est_r']  # Remove old data if it exists
        h5_file.create_dataset('est_r', data=depth_predicted)

def batch_process_h5_files(root_folder, encoder='vitl'):
    """
    Batch process all H5 files in the specified folder and save the depth results in the same H5 files.
    """
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    }
    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(f'depth_anything_v2_{encoder}.pth', map_location='cpu'))
    model = model.to(DEVICE).eval()

    # Get all H5 file paths (including subfolders)
    h5_paths = generate_all_h5_paths(root_folder)
    print(f"Found {len(h5_paths)} H5 files.")

    # Iterate over each H5 file
    for h5_file_path in tqdm(h5_paths, desc="Processing H5 Files"):
        try:
            # Read RGB data
            with h5py.File(h5_file_path, 'r') as h5_file:
                if 'rgb' not in h5_file:
                    raise KeyError(f"Key 'rgb' not found in file {h5_file_path}")
                rgb_image = h5_file['rgb'][:]  # Read RGB data

            # Depth estimation
            depth_predicted = process_depth_image(model, rgb_image)

            # Save the depth map back to the H5 file
            write_est_to_h5(h5_file_path, depth_predicted)

        except Exception as e:
            print(f"Error occurred while processing file {h5_file_path}: {e}")

if __name__ == "__main__":
    root_folder = 'train'  # Replace with the path to your folder of H5 files (including subfolders)
    batch_process_h5_files(root_folder)
