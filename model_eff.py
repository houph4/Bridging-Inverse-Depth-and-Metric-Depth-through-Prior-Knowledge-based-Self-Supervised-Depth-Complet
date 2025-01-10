import torch
import torch.nn as nn
import torch.nn.functional as F
from frequfusion import FreqFusion
from rgbmodule.rgbextractor.efficientvit.seg import efficientvit_depths_l0
from resnet_cbam import HAM

# Encoder that extracts features from sparse depth, RGB image, and estimated depth
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.sparse_depth_extractor = efficientvit_depths_l0()
        self.estimated_depth_extractor = efficientvit_depths_l0()

        # FreqFusion module for merging high and low resolution features
        self.fusion_stage4 = HAM(num_branches=2, channels=512)
        self.fusion_stage3 = HAM(num_branches=2, channels=256)
        self.fusion_stage2 = HAM(num_branches=2, channels=128)
        self.fusion_stage1 = HAM(num_branches=2, channels=64)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, sparse_depth, estimated_depth):
        """
        Forward pass of the Encoder.

        Args:
            sparse_depth (torch.Tensor): Input tensor of sparse depth.
            estimated_depth (torch.Tensor): Input tensor of estimated depth.

        Returns:
            tuple: Merged features from different fusion stages.
                - xpp_stage4 (torch.Tensor): Features from fusion stage 4.
                - xpp_stage3 (torch.Tensor): Features from fusion stage 3.
                - xpp_stage2 (torch.Tensor): Features from fusion stage 2.
                - xpp_stage1 (torch.Tensor): Features from fusion stage 1.
        """
        # Extract sparse depth features
        xp_stage4, xp_stage3, xp_stage2, xp_stage1, xp_stage0 = self.sparse_depth_extractor(sparse_depth)

        # Extract estimated depth features
        xe_stage4, xe_stage3, xe_stage2, xe_stage1, xe_stage0 = self.estimated_depth_extractor(estimated_depth)

        # Merge features from the respective fusion stages
        xpp_stage4 = self.fusion_stage4(xp_stage4, xe_stage4)
        xpp_stage3 = self.fusion_stage3(xp_stage3, xe_stage3)
        xpp_stage2 = self.fusion_stage2(xp_stage2, xe_stage2)
        xpp_stage1 = self.fusion_stage1(xp_stage1, xe_stage1)

        return xpp_stage4, xpp_stage3, xpp_stage2, xpp_stage1



# Decoder to upsample the concatenated features to depth map
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        # To ensure positive uncertainty values
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1)
        )

        self.conv4 = nn.Conv2d(64, 64, kernel_size=1)

        self.ff1 = FreqFusion(hr_channels=64, lr_channels=64)
        self.ff2 = FreqFusion(hr_channels=64, lr_channels=128)
        self.ff3 = FreqFusion(hr_channels=64, lr_channels=192)

        self.depth_branch = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=5, stride=1, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(128, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(32, 1, kernel_size=5, stride=1, padding=2, bias=True),

        )
        #
        self.uncertainty_branch =  nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=5, stride=1, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(128, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(32, 1, kernel_size=5, stride=1, padding=2, bias=True),
            # nn.Sigmoid()
        )

    def _concat(self, fd, fe, dim=1):
        # Decoder feature may have additional padding

        f = torch.cat((fd, fe), dim=dim)

        return f

    def forward(self, xpp_final,xpp_stage3,xpp_stage2,xpp_stage1):

        x1 = self.conv4(xpp_stage1)
        x2 = self.conv3(xpp_stage2)
        x3 = self.conv2(xpp_stage3)
        x4 = self.conv1(xpp_final)

        _, x3, x4_up = self.ff1(hr_feat=x3, lr_feat=x4)
        _, x2, x34_up = self.ff2(hr_feat=x2, lr_feat=torch.cat([x3, x4_up],dim=1))
        _, x1, x234_up = self.ff3(hr_feat=x1, lr_feat=torch.cat([x2, x34_up],dim=1))
        x1234 = torch.cat([x1, x234_up],dim=1)  # channel=4c, 1/4 img size

        depth = self.depth_branch(x1234)
        uncertainty_map = self.uncertainty_branch(x1234)

        return depth,  uncertainty_map


class DiffusionModel(nn.Module):
    def __init__(self):
        super(DiffusionModel, self).__init__()
        self.encoder = Encoder()
        self.undecoder = Decoder()

    def forward(self,  sparse_depth, estimated_depth):
        xpp_stage4,xpp_stage3,xpp_stage2,xpp_stage1= self.encoder(sparse_depth, estimated_depth)
        output_depth, uncertainty = self.undecoder(xpp_stage4, xpp_stage3, xpp_stage2, xpp_stage1)
        output_depth = F.interpolate(output_depth, size=(256, 320), mode='bilinear', align_corners=False)
        uncertainty = F.interpolate(uncertainty, size=(256, 320), mode='bilinear', align_corners=False)
        # output_depth = F.interpolate(output_depth, size=(480, 640), mode='bilinear', align_corners=False)
        # uncertainty = F.interpolate(uncertainty, size=(480, 640), mode='bilinear', align_corners=False)
        # output_depth = F.interpolate(output_depth, size=(256, 1216), mode='bilinear', align_corners=False)
        # uncertainty = F.interpolate(uncertainty, size=(256, 1216), mode='bilinear', align_corners=False)
        return output_depth,uncertainty

