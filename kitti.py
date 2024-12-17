import os
import cv2
import torch
from da2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2
from tqdm import tqdm
import glob
import numpy as np

def load_depth_image(depth_image_path):
    # Load a 16-bit unsigned integer depth image and convert it to meters
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
    if depth_image is None:
        raise FileNotFoundError(f"Unable to load depth image, please check the path: {depth_image_path}")

    # Convert depth values from millimeters to meters (assuming KITTI dataset where depth values are stored in millimeters)
    depth_image_in_meters = depth_image.astype(np.float32) / 256.0

    return depth_image_in_meters

def generate_depth_paths(rgb_root_folder, depth_folder):
    rgb_paths = []
    depth_paths = []

    # Traverse all drive folders under the root directory
    drive_folders = sorted(glob.glob(os.path.join(rgb_root_folder, '2011_09_30_drive_*_sync')))
    for drive_folder in drive_folders:
        # Only process image_02 and image_03 folders
        for image_folder in ['image_02', 'image_03']:
            image_path = os.path.join(drive_folder, image_folder, 'data')
            if not os.path.exists(image_path):
                continue

            rgb_images = sorted(glob.glob(os.path.join(image_path, '*.png')))
            for rgb_path in rgb_images:
                rgb_filename = os.path.basename(rgb_path)
                sequence_number = int(rgb_filename.split('.')[0])
                drive_name = os.path.basename(drive_folder)
                depth_image_number = image_folder  # Since we only process image_02 and image_03

                # Construct the corresponding depth image path
                depth_path = os.path.join(depth_folder, 'train', drive_name, 'proj_depth', 'groundtruth',
                                          depth_image_number, f'{sequence_number:010d}.png')

                # Add corresponding RGB and depth paths to the list only if the depth image path exists
                if os.path.exists(depth_path):
                    rgb_paths.append(rgb_path)
                    depth_paths.append(depth_path)

    return rgb_paths, depth_paths


def process_depth_images(rgb_image_path, real_depth_path, encoder='vits'):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    }

    dataset = 'vkitti'  # 'hypersim' for indoor model, 'vkitti' for outdoor model
    max_depth = 80  # 20 for indoor model, 80 for outdoor model
    model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
    model.load_state_dict(
        torch.load(f'depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location='cpu'))
    model = model.to(DEVICE).eval()

    # Get the resolution of the ground-truth depth map
    real_depth = load_depth_image(real_depth_path)

    # Resize the RGB image to match the resolution of the ground-truth depth map
    raw_img = cv2.imread(rgb_image_path)
    if raw_img is None:
        raise FileNotFoundError("Unable to read the input image, please check the file path.")

    # Perform inference to predict the depth map
    depth_predicted = model.infer_image(raw_img).squeeze()

    return real_depth, depth_predicted

def batch_process_depth_images(rgb_root_folder, depth_folder, output_root_folder, encoder='vits'):
    rgb_paths, depth_paths = generate_depth_paths(rgb_root_folder, depth_folder)

    for rgb_image_path, real_depth_path in tqdm(zip(rgb_paths, depth_paths), total=len(rgb_paths),
                                                desc="Processing Images"):
        real_depth, depth_predicted = process_depth_images(rgb_image_path, real_depth_path, encoder)

        # Construct the output path
        drive_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(real_depth_path)))))
        depth_image_number = os.path.basename(os.path.dirname(real_depth_path))

        output_dir = os.path.join(output_root_folder, 'train', drive_name, depth_image_number)
        os.makedirs(output_dir, exist_ok=True)

        # Get the file name and save the predicted depth map
        base_name = os.path.basename(real_depth_path)

        cv2.imwrite(os.path.join(output_dir, base_name), depth_predicted)


if __name__ == "__main__":
    rgb_root_folder = 'raw/2011_09_30'  # Root folder of the RGB images
    depth_folder = 'data_depth_annotated'  # Root folder of the depth maps
    output_root_folder = 'depth_prediction'  # Root folder for saving predicted depth maps

    batch_process_depth_images(rgb_root_folder, depth_folder, output_root_folder)
