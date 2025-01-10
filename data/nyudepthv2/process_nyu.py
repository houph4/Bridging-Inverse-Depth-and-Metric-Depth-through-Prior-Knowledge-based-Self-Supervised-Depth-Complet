import os
import cv2
import torch
import h5py
import numpy as np
from da2.depth_anything_v2.dpt import DepthAnythingV2
from tqdm import tqdm

def generate_all_h5_paths(root_folder):
    """
    遍历主文件夹及其所有子文件夹，生成所有 H5 文件的路径。
    """
    h5_paths = []
    for dirpath, _, filenames in os.walk(root_folder):
        for file in filenames:
            if file.endswith('.h5'):
                h5_paths.append(os.path.join(dirpath, file))
    return sorted(h5_paths)

def process_depth_image(model, rgb_image):
    """
    对单个 RGB 图像进行深度估计
    """
    rgb_image = np.transpose(rgb_image, (1, 2, 0))  # 转换为 (H, W, C)
    raw_img_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)  # 转换为 BGR 顺序

    with torch.no_grad():
        depth_predicted = model.infer_image(raw_img_rgb).squeeze()
    return depth_predicted

def write_est_to_h5(h5_file_path, depth_predicted):
    """
    将估计的深度图写入 H5 文件中，键名为 'est'
    """
    with h5py.File(h5_file_path, 'r+') as h5_file:
        if 'est_r' in h5_file:
            del h5_file['est_r']  # 如果已存在，删除旧数据
        h5_file.create_dataset('est_r', data=depth_predicted)

def batch_process_h5_files(root_folder, encoder='vitl'):
    """
    批量处理文件夹内所有 H5 文件中的 RGB 数据，并将深度结果保存到同一个 H5 文件中
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

    # 获取所有 H5 文件的路径（包括子文件夹）
    h5_paths = generate_all_h5_paths(root_folder)
    print(f"共找到 {len(h5_paths)} 个 H5 文件。")

    # 遍历每个 H5 文件
    for h5_file_path in tqdm(h5_paths, desc="Processing H5 Files"):
        try:
            # 读取 RGB 数据
            with h5py.File(h5_file_path, 'r') as h5_file:
                if 'rgb' not in h5_file:
                    raise KeyError(f"文件 {h5_file_path} 中未找到键 'rgb'")
                rgb_image = h5_file['rgb'][:]  # 读取 RGB 数据

            # 深度估计
            depth_predicted = process_depth_image(model, rgb_image)

            # 将深度图保存回 H5 文件
            write_est_to_h5(h5_file_path, depth_predicted)

        except Exception as e:
            print(f"处理文件 {h5_file_path} 时发生错误: {e}")

if __name__ == "__main__":
    root_folder = 'train'  # 替换为你的 H5 文件夹路径（包含子文件夹）
    batch_process_h5_files(root_folder)

