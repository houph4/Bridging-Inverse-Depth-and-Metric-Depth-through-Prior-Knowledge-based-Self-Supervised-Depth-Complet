import os
import numpy as np
import os.path as osp
import cv2
import torchvision.transforms as T
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image
__all__ = [
    'VOID',
]


def load_depth(path):
    '''
    Loads a depth map from a 16-bit PNG file

    Args:
    path : str
      path to 16-bit PNG file

    Returns:
    numpy : depth map
    '''
    # Loads depth map from 16-bit PNG file
    z = np.array(Image.open(path), dtype=np.float32)
    # Assert 16-bit (not 8-bit) depth map
    z = z/256.0
    z[z <= 0] = 0.0
    return z


def read_gen(file_name, pil=False):
    ext = osp.splitext(file_name)[-1]
    if ext in ['.png', '.jpeg', '.ppm', '.jpg']:
        img = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)  
        if img is None:
            raise ValueError(f"Failed to read image from {file_name}")
        return img
    elif ext in ['.bin', '.raw']:
        return np.load(file_name)
    return []

class BaseDataset(Dataset):
    def __init__(self, mode):
        self.mode = mode

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    # A workaround for a pytorch bug
    # https://github.com/pytorch/vision/issues/2194
    class ToNumpy:
        def __call__(self, sample):
            return np.array(sample)


class VOID(BaseDataset):
    def __init__(self, mode):
        super(VOID, self).__init__(mode)

        self.mode = mode
        datapath = 'data/void/void_release/void_1500'
        parent_dir = os.path.dirname(datapath)

        self.image_list = []
        self.extra_info = []
        height, width = (240, 320)
        crop_size = (228, 304)

        self.height = height
        self.width = width
        self.crop_size = crop_size

        # appendix = "train" if mode == "train" else "test"
        appendix = mode

        # Glue code between persefone data and my shitty format
        intrinsics_txt = open(osp.join(datapath, f"{appendix}_intrinsics.txt"), 'r')
        rgb_txt = open(osp.join(datapath, f"{appendix}_image.txt"), 'r')
        hints_txt = open(osp.join(datapath, f"{appendix}_sparse_depth.txt"), 'r')
        gt_txt = open(osp.join(datapath, f"{appendix}_ground_truth.txt"), 'r')
        valid_txt = open(osp.join(datapath, f"{appendix}_validity_map.txt"))
        pred_txt = open(osp.join(datapath, f"{appendix}_pred.txt"))

        while True:
            i_path = intrinsics_txt.readline().strip()
            rgb_path = rgb_txt.readline().strip()
            hints_path = hints_txt.readline().strip()
            gt_path = gt_txt.readline().strip()
            valid_path = valid_txt.readline().strip()
            pred_path = pred_txt.readline().strip()

            if not i_path or not rgb_path or not hints_path or not gt_path or not valid_path:
                break

            self.image_list += [[osp.join(parent_dir, i_path),
                                 osp.join(parent_dir, rgb_path),
                                 osp.join(parent_dir, hints_path),
                                 osp.join(parent_dir, gt_path),
                                 osp.join(parent_dir, valid_path),
                                 osp.join(parent_dir, pred_path)]]
            self.extra_info += [[rgb_path.split('/')[-1]]]

        intrinsics_txt.close()
        rgb_txt.close()
        hints_txt.close()
        gt_txt.close()
        valid_txt.close()
        pred_txt.close()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        K = np.loadtxt(self.image_list[index][0])
        rgb = read_gen(self.image_list[index][1])
        hints_depth = load_depth(self.image_list[index][2])
        gt_depth = load_depth(self.image_list[index][3])
        pred = load_depth(self.image_list[index][5])
        # depth_image_path= self.image_list[index][5]
        # pred = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
        # pred = pred.astype(np.float32)

        rgb = Image.fromarray(rgb, mode='RGB')
        dep_sp = Image.fromarray(hints_depth, mode='F')
        dep = Image.fromarray(gt_depth, mode='F')
        dep_pred = Image.fromarray(pred, mode='F')

        if self.mode == 'train':
            _scale = np.random.uniform(1.0, 1.5)
            degree = np.random.uniform(-5.0, 5.0)
            flip = np.random.uniform(0.0, 1.0)
            scale = int(self.height * _scale)

            if flip > 0.5:
                rgb = TF.hflip(rgb)
                dep_sp =TF.hflip(dep_sp)
                dep_pred = TF.hflip(dep_pred)

            rgb = TF.rotate(rgb, angle=degree)
            dep_sp  = TF.rotate(dep_sp, angle=degree)
            dep_pred = TF.rotate(dep_pred, angle=degree)

            t_rgb = T.Compose([
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                T.CenterCrop(self.crop_size),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

            t_dep = T.Compose([
                T.CenterCrop(self.crop_size),
                self.ToNumpy(),
                T.ToTensor()
            ])

            t_est = T.Compose([
                T.CenterCrop(self.crop_size),
                self.ToNumpy(),
                T.ToTensor()
            ])

            rgb = t_rgb(rgb)
            dep_sp = t_dep(dep_sp)
            dep_pred = t_est(dep_pred)
            dep = t_dep(dep)

        else:
            t_rgb = T.Compose([
                T.CenterCrop(self.crop_size),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

            t_dep = T.Compose([
                T.CenterCrop(self.crop_size),
                self.ToNumpy(),
                T.ToTensor()
            ])

            t_est = T.Compose([
                T.CenterCrop(self.crop_size),
                self.ToNumpy(),
                T.ToTensor()
            ])

            rgb = t_rgb(rgb)
            dep_sp = t_dep(dep_sp)
            dep_pred = t_est(dep_pred)
            dep = t_dep(dep)

        rgb = TF.pad(rgb, padding=[8, 14], padding_mode='edge')
        #torch.Size([3, 228, 304])  torch.Size([3, 256, 320])
        dep_sp = TF.pad(dep_sp, padding=[8, 14], padding_mode='constant')
        #dep_sp_b torch.Size([1, 228, 304]) , dep_sp_a torch.Size([1, 256, 320])
        dep = TF.pad(dep, padding=[8, 14], padding_mode='constant')
        #torch.Size([1, 228, 304]) torch.Size([1, 256, 320])
        dep_pred = TF.pad(dep_pred,padding=[8, 14], padding_mode='constant')
        #est torch.Size([1, 228, 304])  est_a torch.Size([1, 256, 320])
        return rgb, dep_sp, dep_pred, dep
