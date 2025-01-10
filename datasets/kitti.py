import random
import os
import numpy as np
import glob
from PIL import Image
import torch
import augs
import cv2
from torchvision import transforms
from PIL import Image

__all__ = [
    "KITTI"
]

def get_data_transforms(mode='train', height=256, width=1216):
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((height, width)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor()
        ])
    else:
        return transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToTensor()
        ])

class KITTI(torch.utils.data.Dataset):
    """
    kitti depth completion dataset: http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion
    """

    def __init__(self, path='datas/kitti', mode='train', height=256, width=1216, mean=(90.9950, 96.2278, 94.3213),
                 std=(79.2382, 80.5267, 82.1483), RandCrop=False, tp_min=50, *args, **kwargs):
        self.base_dir = path
        self.height = height
        self.width = width
        self.mode = mode
        if mode == 'train':
            self.transform = augs.Compose([
                augs.Jitter(),
                augs.Flip(),
                augs.Norm(mean=mean, std=std),
            ])
        else:
            self.transform = augs.Compose([
                augs.Norm(mean=mean, std=std),
            ])
        self.RandCrop = RandCrop and mode == 'train'
        self.tp_min = tp_min
        # self.transform = get_data_transforms(mode=mode, height=height, width=width)
        if mode in ['train', 'val']:
            self.depth_path = os.path.join(self.base_dir, 'data_depth_annotated', mode)
            self.lidar_path = os.path.join(self.base_dir, 'depth_velodyne', mode)
            self.metric_preditction_path = os. path.join(self.base_dir,'depth_prediction',mode)
            self.relative_preditction_path = os.path.join(self.base_dir, 'depth_prediction_relative', mode)
            self.depths = list(sorted(glob.iglob(self.depth_path + "/**/*.png", recursive=True)))
            self.lidars = list(sorted(glob.iglob(self.lidar_path + "/**/*.png", recursive=True)))
            self.metric_depths = list(sorted(glob.iglob(self.metric_preditction_path + "/**/*.png", recursive=True)))
            self.relative_depths = list(sorted(glob.iglob(self.relative_preditction_path + "/**/*.png", recursive=True)))
        elif mode == 'selval':
            self.depth_path = os.path.join(self.base_dir, 'val_selection_cropped', 'groundtruth_depth')
            self.lidar_path = os.path.join(self.base_dir, 'val_selection_cropped', 'velodyne_raw')
            self.image_path = os.path.join(self.base_dir, 'val_selection_cropped', 'image')
            self.metric_preditction_path = os. path.join(self.base_dir,'depth_prediction',mode)
            self.relative_preditction_path = os.path.join(self.base_dir, 'depth_prediction_relative', mode)
            self.depths = list(sorted(glob.iglob(self.depth_path + "/*.png", recursive=True)))
            self.lidars = list(sorted(glob.iglob(self.lidar_path + "/*.png", recursive=True)))
            self.images = list(sorted(glob.iglob(self.image_path + "/*.png", recursive=True)))
            self.metric_depths = list(sorted(glob.iglob(self.metric_preditction_path + "/*.png", recursive=True)))
            self.relative_depths = list(sorted(glob.iglob(self.relative_preditction_path + "/*.png", recursive=True)))
        elif mode == 'test':
            self.lidar_path = os.path.join(self.base_dir, 'test_depth_completion_anonymous', 'velodyne_raw')
            self.image_path = os.path.join(self.base_dir, 'test_depth_completion_anonymous', 'image')
            self.metric_preditction_path = os. path.join(self.base_dir,'depth_prediction',mode)
            self.relative_preditction_path = os.path.join(self.base_dir, 'depth_prediction_relative', mode)
            self.lidars = list(sorted(glob.iglob(self.lidar_path + "/*.png", recursive=True)))
            self.images = list(sorted(glob.iglob(self.image_path + "/*.png", recursive=True)))
            self.metric_depths = list(sorted(glob.iglob(self.metric_preditction_path + "/*.png", recursive=True)))
            self.relative_depths = list(sorted(glob.iglob(self.relative_preditction_path + "/*.png", recursive=True)))
            self.depths = self.lidars
        else:
            raise ValueError("Unknown mode: {}".format(mode))
        assert (len(self.depths) == len(self.lidars))
        self.names = [os.path.split(path)[-1] for path in self.depths]
        x = np.arange(width)
        y = np.arange(height)
        xx, yy = np.meshgrid(x, y)
        xy = np.stack((xx, yy), axis=-1)
        self.xy = xy

    def __len__(self):
        return len(self.depths)

    def __getitem__(self, index):
        return self.get_item(index)

    def get_item(self, index):
        depth = self.load_depth_image(self.depths[index])
        depth = np.expand_dims(depth, axis=2)
        lidar = self.load_depth_image(self.lidars[index])
        lidar = np.expand_dims(lidar, axis=2)
        metric_depth = self.load_depth_image(self.relative_depths[index])
        # metric_depth = cv2.imread(self.metric_depths[index],cv2.IMREAD_UNCHANGED)
        metric_depth = np.expand_dims(metric_depth,axis=2)
        file_names = self.depths[index].split('/')
        if self.mode in ['train', 'val']:
            rgb_path = os.path.join(*file_names[:-7], 'raw', file_names[-5].split('_drive')[0], file_names[-5],
                                    file_names[-2], 'data', file_names[-1])
        elif self.mode in ['selval', 'test']:
            rgb_path = self.images[index]
        else:
            raise ValueError("Unknown mode: {}".format(self.mode))
        # rgb = Image.open(rgb_path).convert('RGB')
        rgb = self.pull_RGB(rgb_path)
        rgb = rgb.astype(np.float32)
        lidar = lidar.astype(np.float32)
        depth = depth.astype(np.float32)
        metric_depth = metric_depth.astype(np.float32)
        if self.transform:
            rgb, lidar, depth, metric_depth = self.transform(rgb, lidar, depth, metric_depth)
        rgb = rgb.transpose(2, 0, 1).astype(np.float32)
        lidar = lidar.transpose(2, 0, 1).astype(np.float32)
        # lidar = np.squeeze(lidar, axis=0)
        depth = depth.transpose(2, 0, 1).astype(np.float32)
        # depth = np.squeeze(depth, axis=0)
        metric_depth = metric_depth.transpose(2, 0, 1).astype(np.float32)
        # metric_depth = np.squeeze(metric_depth, axis=0)
        # lidar = cv2.resize(lidar, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        # depth = cv2.resize(depth, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        # metric_depth = cv2.resize(metric_depth, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        # lidar = np.expand_dims(lidar, axis=0)
        # depth  = np.expand_dims(depth , axis=0)
        # metric_depth = np.expand_dims(metric_depth, axis=0)

        tp = rgb.shape[1] - self.height
        lp = (rgb.shape[2] - self.width) // 2
        if self.RandCrop and self.mode == 'train':
            tp = random.randint(self.tp_min, tp)
            lp = random.randint(0, rgb.shape[2] - self.width)
        rgb = rgb[:, tp:tp + self.height, lp:lp + self.width]
        lidar = lidar[:, tp:tp + self.height, lp:lp + self.width]
        depth = depth[:, tp:tp + self.height, lp:lp + self.width]
        metric_depth = metric_depth [:, tp:tp + self.height, lp:lp + self.width]
        return rgb, lidar, metric_depth, depth

    def pull_RGB(self, path):
        img = np.array(Image.open(path).convert('RGB'), dtype=np.uint8)
        return img

    # def pull_DEPTH(self, path):
    #     depth_png = np.array(Image.open(path), dtype=int)
    #     assert (np.max(depth_png) > 255)
    #     depth_image = (depth_png / 256.).astype(np.float32)
    #     return depth_image

    # def pull_MDEPTH(self, path):
    #     """
    #     Load and process a Metric Depth map file.
    #
    #     Parameters:
    #     - path (str): Path to the Metric Depth image file.
    #
    #     Returns:
    #     - depth_image (np.ndarray): Processed Metric Depth image as a float32 array.
    #     """
    #     depth_png = np.array(Image.open(path), dtype=np.float32)  # 保持为浮点类型
    #     if np.max(depth_png) > 80 or np.min(depth_png) < 0:
    #         raise ValueError(
    #             f"Metric Depth image at {path} has invalid values. "
    #             f"Expected range [0, 80], got min: {np.min(depth_png)}, max: {np.max(depth_png)}."
    #         )
    #     depth_image = (depth_png / 80.).astype(np.float32) # 归一化，与 DEPTH 对齐（最大值不同）
    #     return depth_image

    def load_depth_image(self,depth_image_path):
        # 加载 16 位无符号整数格式的深度图像，并将其转换为米
        depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
        if depth_image is None:
            raise FileNotFoundError(f"无法加载深度图像，请检查路径: {depth_image_path}")

        # 将深度值从毫米转换为米（假设 KITTI 数据集，深度值以毫米为单位存储）
        depth_image_in_meters = depth_image.astype(np.float32) / 256.0

        return depth_image_in_meters
