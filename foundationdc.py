import torch.nn as nn
import torch
from model_eff import DiffusionModel
from jet import knn_outlier_removal_multiscale
from scaler import compute_scale_and_shift_theil_sen
import numpy as np

class DepthAlignmentWrapper(nn.Module):
    def __init__(self, knn_model, sample_size=200, inlier_threshold=0.6, k_size=5, threshold=0.01, num_scales=1):
        """
        A PyTorch Module to handle KNN outlier removal and affine alignment.

        Args:
            knn_model: Function for KNN outlier removal (e.g., knn_outlier_removal_multiscale).
            sample_size: Number of samples for affine alignment.
            inlier_threshold: Threshold for inlier ratio during affine alignment.
            k_size: Kernel size for KNN outlier removal.
            threshold: Threshold for KNN outlier detection.
            num_scales: Number of scales for multi-scale KNN.
        """
        super(DepthAlignmentWrapper, self).__init__()
        self.knn_model = knn_model
        self.sample_size = sample_size
        self.inlier_threshold = inlier_threshold
        self.k_size = k_size
        self.threshold = threshold
        self.num_scales = num_scales

    def forward(self, depth_predicted, sparse_depth, device):
        """
        Perform KNN outlier removal and affine alignment on depth predictions.

        Args:
            depth_predicted (torch.Tensor): Predicted depth map (H, W) or (1, H, W).
            sparse_depth (torch.Tensor or np.ndarray): Sparse real depth map.
            device (torch.device): Device to use for computation.

        Returns:
            depth_predicted_aligned (torch.Tensor): Aligned predicted depth map.
            inlier_ratio (float): Ratio of inliers used for affine alignment.
        """
        # Step 1: Ensure sparse_depth is a tensor
        if isinstance(sparse_depth, np.ndarray):
            sparse_depth_tensor = torch.tensor(sparse_depth).unsqueeze(0).unsqueeze(0).float().to(device)
        else:
            sparse_depth_tensor = sparse_depth


        # Step 2: Apply KNN outlier removal
        _, knn_valid_mask = self.knn_model(sparse_depth_tensor, k_size=self.k_size, threshold=self.threshold, num_scales=self.num_scales)
        knn_valid_mask_np = knn_valid_mask.squeeze().cpu().numpy().astype(bool)

        # Step 3: Prepare depth_predicted for alignment
        depth_predicted = depth_predicted.squeeze().cpu().numpy()

        # Step 4: Compute affine alignment (scale and shift)
        scale, shift, inlier_ratio = compute_scale_and_shift_theil_sen(
            prediction=depth_predicted,
            target=sparse_depth_tensor.squeeze().cpu().numpy(),
            mask=knn_valid_mask_np,
            sample_size=self.sample_size,
            inlier_threshold=self.inlier_threshold
        )

        # Step 5: Apply affine alignment
        depth_predicted_aligned = depth_predicted * scale + shift

        # Step 6: Clamp aligned depth to valid range of sparse_depth
        real_depth_min = float(sparse_depth_tensor.min())
        real_depth_max = float(sparse_depth_tensor.max())
        depth_predicted_aligned = torch.tensor(depth_predicted_aligned).float().to(device)
        depth_predicted_aligned = torch.clamp(depth_predicted_aligned, real_depth_min, real_depth_max)

        # Add batch and channel dimensions
        depth_predicted_aligned = depth_predicted_aligned.clone().detach().unsqueeze(0).float().to(device)
        depth_predicted_aligned = depth_predicted_aligned.transpose(0, 1)

        return depth_predicted_aligned, inlier_ratio


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.RESnet = DiffusionModel()
        self.min_predict_depth = 0.1
        self.max_predict_depth = 5.0
    def forward(self, depth, est):
        # Res = self.RESnet(est, depth, rgb)
        Res,uncertain = self.RESnet(depth, est)
        out = Res+est
        # out = torch.sigmoid(out)
        # out = \
        #     self.min_predict_depth / (out + self.min_predict_depth / self.max_predict_depth)
        return out,uncertain
        # return Res, out, uncertain


class Foundationdc(nn.Module):
    def __init__(self):
        super(Foundationdc, self).__init__()
        self.resnet = Net()
        self.alignment = DepthAlignmentWrapper(
            knn_outlier_removal_multiscale,
            sample_size=20,
            inlier_threshold=1.0,
            k_size=3,
            threshold=1,
            num_scales=1
        )

    def forward(self, sparse_depth, estimated_depth):

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        aligned, inlier_ratio = self.alignment(
            estimated_depth, sparse_depth, device
        )

        if aligned.size() == torch.Size([256, 1, 320]):
            # Permute dimensions to [1, 1, 256, 1216]
            aligned = aligned.unsqueeze(0).permute(0,2,1,3)

        out,uncertain = self.resnet(sparse_depth, aligned)

        return out,uncertain



