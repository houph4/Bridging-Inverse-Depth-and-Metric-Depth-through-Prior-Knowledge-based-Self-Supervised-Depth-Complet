import os
import cv2
import torch
import numpy as np
from da2.depth_anything_v2.dpt import DepthAnythingV2
from tqdm import tqdm

def generate_all_rgb_paths(root_folder):
    """
    Traverse the main folder and all subfolders under 'image' folders to generate paths for all RGB images.
    """
    rgb_paths = []
    # Retrieve all category folders
    categories = [category for category in os.listdir(root_folder) if
                  os.path.isdir(os.path.join(root_folder, category))]

    # Use tqdm to track progress
    for category in tqdm(categories, desc="Finding Image Categories"):
        category_path = os.path.join(root_folder, category)
        # Construct the path for the image folder
        image_folder_path = os.path.join(category_path, "image")
        # Check if the image folder exists
        if os.path.isdir(image_folder_path):
            for dirpath, _, filenames in os.walk(image_folder_path):
                for file in filenames:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Support common image formats
                        rgb_paths.append(os.path.join(dirpath, file))
    return sorted(rgb_paths)

def save_depth_image(depth_image, save_path):
    """
    Save the depth image as an image file.
    """
    depth_image = (depth_image * 255).astype(np.uint8)  # Scale data (assuming depth values are in the [0, 1] range)
    cv2.imwrite(save_path, depth_image)

def batch_process_rgb_images(root_folder, save_folder, encoder='vitl'):
    """
    Batch process all RGB images in the specified folder and save the depth results to the designated folder.
    """
    # Get all RGB image paths (including subfolders)
    rgb_paths = generate_all_rgb_paths(root_folder)
    rgb_images = [cv2.imread(path) for path in tqdm(rgb_paths, desc="Loading RGB Images")]  # Read all images
    print(f"Found {len(rgb_images)} RGB images.")  # Output the number of found images

    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    }
    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(f'depth_anything_v2_{encoder}.pth', map_location='cpu'))
    model = model.to(DEVICE).eval()

    # Process each RGB image
    for rgb_image, rgb_file_path in tqdm(zip(rgb_images, rgb_paths), desc="Processing RGB Images",
                                         total=len(rgb_images)):
        try:
            # Convert to BGR format
            raw_img_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

            with torch.no_grad():
                depth_predicted = model.infer_image(raw_img_rgb).squeeze()

            # Construct the path to save the depth map
            # Get the category path (omit the last 'image' part)
            category_path = os.path.dirname(os.path.dirname(rgb_file_path))
            # Create the pred_depth folder path
            pred_depth_folder = os.path.join(save_folder, os.path.basename(category_path), 'pred_depth')
            os.makedirs(pred_depth_folder, exist_ok=True)

            # Save the depth map
            depth_save_path = os.path.join(pred_depth_folder,
                                           os.path.basename(rgb_file_path).replace('.jpg', '_depth.png').replace(
                                               '.jpeg', '_depth.png').replace('.png', '_depth.png'))
            save_depth_image(depth_predicted, depth_save_path)

        except Exception as e:
            print(f"Error occurred while processing file {rgb_file_path}: {e}")

if __name__ == "__main__":
    root_folder = 'void_release/void_150/data'  # Main folder path containing category folders
    save_folder = 'void_release/void_150/data'  # Main path to save depth maps, same as root_folder
    batch_process_rgb_images(root_folder, save_folder)
