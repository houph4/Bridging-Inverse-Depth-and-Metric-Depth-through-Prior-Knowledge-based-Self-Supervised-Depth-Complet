import os
import cv2
import torch
import numpy as np
from da2.depth_anything_v2.dpt import DepthAnythingV2
from tqdm import tqdm


def generate_all_rgb_paths(root_folder):
    """
    遍历主文件夹及其所有子文件夹的 'image' 文件夹，生成所有 RGB 图像的路径。
    """
    rgb_paths = []
    # 获取所有类别文件夹
    categories = [category for category in os.listdir(root_folder) if
                  os.path.isdir(os.path.join(root_folder, category))]

    # 使用 tqdm 追踪进度
    for category in tqdm(categories, desc="Finding Image Categories"):
        category_path = os.path.join(root_folder, category)
        # 构建 image 文件夹的路径
        image_folder_path = os.path.join(category_path, "image")
        # 检查 image 文件夹是否存在
        if os.path.isdir(image_folder_path):
            for dirpath, _, filenames in os.walk(image_folder_path):
                for file in filenames:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # 适应常见图像格式
                        rgb_paths.append(os.path.join(dirpath, file))
    return sorted(rgb_paths)


def save_depth_image(depth_image, save_path):
    """
    保存深度图为图像文件
    """
    depth_image = (depth_image * 255).astype(np.uint8)  # 数据缩放（假设深度值在 [0, 1] 范围内）
    cv2.imwrite(save_path, depth_image)


def batch_process_rgb_images(root_folder, save_folder, encoder='vitl'):
    """
    批量处理文件夹内所有 RGB 图像，并将深度结果保存到指定的文件夹中
    """
    # 获取所有 RGB 图像的路径（包括子文件夹）
    rgb_paths = generate_all_rgb_paths(root_folder)
    rgb_images = [cv2.imread(path) for path in tqdm(rgb_paths, desc="Loading RGB Images")]  # 读取所有图像
    print(f"共找到 {len(rgb_images)} 个 RGB 图像。")  # 输出找到的图像数量

    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    }
    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(f'depth_anything_v2_{encoder}.pth', map_location='cpu'))
    model = model.to(DEVICE).eval()

    # 遍历处理每个 RGB 图像
    for rgb_image, rgb_file_path in tqdm(zip(rgb_images, rgb_paths), desc="Processing RGB Images",
                                         total=len(rgb_images)):
        try:
            # 转换为 BGR 顺序
            raw_img_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

            with torch.no_grad():
                depth_predicted = model.infer_image(raw_img_rgb).squeeze()

            # 构建保存深度图的路径
            # 获取类别路径（去掉最后的 'image' 部分）
            category_path = os.path.dirname(os.path.dirname(rgb_file_path))
            # 创建 pred_depth 文件夹路径
            pred_depth_folder = os.path.join(save_folder, os.path.basename(category_path), 'pred_depth')
            os.makedirs(pred_depth_folder, exist_ok=True)

            # 保存深度图
            depth_save_path = os.path.join(pred_depth_folder,
                                           os.path.basename(rgb_file_path).replace('.jpg', '.png').replace(
                                               '.jpeg', '.png'))
            cv2.imwrite(depth_save_path, depth_predicted)

        except Exception as e:
            print(f"处理文件 {rgb_file_path} 时发生错误: {e}")


if __name__ == "__main__":
    root_folder = 'void_release/void_150/data'  # 这里是包含类别文件夹的主文件夹路径
    save_folder = 'void_release/void_150/data'  # 保存深度图的主路径，与 root_folder 相同
    batch_process_rgb_images(root_folder, save_folder)
