import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class HAM(nn.Module):
    def __init__(self, num_branches, channels, curvature=1.0):
        """
        Attention-enhanced Hyperbolic Affinity Module (AttentionHAM)

        Args:
            num_branches (int): Number of input branches.
            channels (int): Number of channels in each branch.
            curvature (float): Curvature value for hyperbolic space (default is 1.0).
        """
        super(HAM, self).__init__()
        self.num_branches = num_branches
        self.channels = channels
        self.curvature = curvature

        # Channel attention module
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # Projection module
        self.projections = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size=1, bias=False)
            for _ in range(num_branches)
        ])

        # Hyperbolic convolution
        self.hyperbolic_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

        # Final fusion layer
        self.fusion_layer = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def apply_attention(self, x):
        """Apply channel attention mechanism."""
        attention = self.channel_attention(x)
        return x * attention

    def hyperbolic_distance(self, u, v, curvature=1.0, eps=1e-6):
        """
        Calculate hyperbolic geodesic distance.

        Args:
            u (torch.Tensor): First tensor.
            v (torch.Tensor): Second tensor.
            curvature (float): Curvature for hyperbolic space.
            eps (float): A small value for numerical stability.
        
        Returns:
            torch.Tensor: Computed distances.
        """
        inner_prod = torch.sum(u * v, dim=1, keepdim=True)
        norm_u = torch.norm(u, dim=1, keepdim=True)
        norm_v = torch.norm(v, dim=1, keepdim=True)

        numerator = 2 * curvature * (norm_u ** 2 + norm_v ** 2 - 2 * inner_prod)
        denominator = (1 - curvature * norm_u ** 2) * (1 - curvature * norm_v ** 2)

        distance = torch.acosh(1 + numerator / denominator.clamp(min=eps))
        return distance

    def forward(self, *inputs):
        """
        Forward propagation.

        Args:
            inputs: Multi-branch input features (such as sparse depth, RGB features, estimated depth, etc.).

        Returns:
            torch.Tensor: Fused features.
        """
        assert len(inputs) == self.num_branches, "Number of input branches must match num_branches!"

        # Project to a unified feature space
        projected_features = [proj(feat) for proj, feat in zip(self.projections, inputs)]

        # Apply channel attention
        projected_features = [self.apply_attention(feat) for feat in projected_features]

        # Fuse branch features
        fused_feature = projected_features[0]
        for i in range(1, self.num_branches):
            dist = self.hyperbolic_distance(fused_feature, projected_features[i], curvature=self.curvature)
            weight = F.softmax(-dist, dim=1)  # Geodesic weight
            fused_feature = fused_feature + weight * projected_features[i]

        # Additional fusion with hyperbolic convolution
        fused_feature = self.hyperbolic_conv(fused_feature)

        # Final feature fusion
        output = self.fusion_layer(fused_feature)
        return output

