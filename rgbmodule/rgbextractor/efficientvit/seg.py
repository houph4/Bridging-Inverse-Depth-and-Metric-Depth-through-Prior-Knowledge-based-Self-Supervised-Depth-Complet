import torch
import torch.nn as nn
from typing import List, Union, Optional
from rgbmodule.rgbextractor.efficientvit.backbone import EfficientViTDepthBackbone
from rgbmodule.rgbextractor.nn import (
    ConvLayer,
    DAGBlock,
    FusedMBConv,
    IdentityLayer,
    MBConv,
    OpSequential,
    ResidualBlock,
    UpSampleLayer,
)
from rgbmodule.rgbextractor.utils import build_kwargs_from_config

__all__ = [
    "EfficientViTDepth",
    "efficientvit_depths_l0",
]



class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))



class SegHead(DAGBlock):
    def __init__(
        self,
        fid_list: List[str],
        in_channel_list: List[int],
        stride_list: List[int],
        head_stride: int,
        head_width: int,
        head_depth: int,
        expand_ratio: float,
        middle_op: str,
        final_expand: Union[float, None],
        n_classes: int,
        dropout: float = 0,
        norm: str = "bn2d",
        act_func: str = "hswish",
    ):
        inputs = {}
        for fid, in_channel, stride in zip(fid_list, in_channel_list, stride_list):
            factor = stride // head_stride
            if factor == 1:
                inputs[fid] = ConvLayer(in_channel, head_width, 1, norm=norm, act_func=None)
            else:
                inputs[fid] = OpSequential(
                    [
                        ConvLayer(in_channel, head_width, 1, norm=norm, act_func=None),
                        UpSampleLayer(factor=factor)
                        # DySample(head_width,factor)
                    ]
                )

        middle = []
        for _ in range(head_depth):
            if middle_op == "mbconv":
                block = MBConv(
                    head_width,
                    head_width,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=(act_func, act_func, None),
                )
            elif middle_op == "fmbconv":
                block = FusedMBConv(
                    head_width,
                    head_width,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=(act_func, None),
                )
            else:
                raise NotImplementedError
            middle.append(ResidualBlock(block, IdentityLayer()))
        middle = OpSequential(middle)

        outputs = {
            "segout": OpSequential(
                [
                    None
                    if final_expand is None
                    else ConvLayer(head_width, head_width * final_expand, 1, norm=norm, act_func=act_func),
                    ConvLayer(
                        head_width * (final_expand or 1),
                        n_classes,
                        1,
                        use_bias=True,
                        dropout=dropout,
                        norm=None,
                        act_func=None,
                    ),

                ]
            )
        }

        super(SegHead, self).__init__(inputs, "add", None, middle=middle, outputs=outputs)



####Key: input, Shape: torch.Size([1, 3, 480, 1280])
#Key: stage0, Shape: torch.Size([1, 3, 240, 640])
# Key: stage1, Shape: torch.Size([1, 16, 120, 320])
# Key: stage2, Shape: torch.Size([1, 32, 60, 160])
# Key: stage3, Shape: torch.Size([1, 64, 30, 80])
# Key: stage4, Shape: torch.Size([1, 128, 15, 40])
# Key: stage_final, Shape: torch.Size([1, 128, 15, 40])
# Key: segout, Shape: torch.Size([1, 128, 60, 160])


class EfficientViTDepth(nn.Module):
    def __init__(self, backbone: EfficientViTDepthBackbone, head: SegHead) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feed_dict = self.backbone(x)
        # feed_dict = self.head(feed_dict)
        x_stage4 = feed_dict["stage4"]
        x_stage3 = feed_dict["stage3"]
        x_stage2 = feed_dict["stage2"]
        x_stage1 = feed_dict["stage1"]
        x_stage0 = feed_dict["stage0"]
        # feed_dicth = self.head(feed_dict)
        # segout = feed_dicth["segout"]
        return x_stage4,x_stage3,x_stage2,x_stage1,x_stage0


def efficientvit_depths_l0(**kwargs) -> EfficientViTDepth:
    from rgbmodule.rgbextractor.efficientvit.backbone import efficientvit_depth_l0

    backbone = efficientvit_depth_l0(**kwargs)

    head = SegHead(
            fid_list=["stage4", "stage3", "stage2"],
            in_channel_list=[128, 64, 32],
            stride_list=[32, 16, 8],
            head_stride=8,
            head_width=32,
            head_depth=1,
            expand_ratio=4,
            middle_op="fmbconv",
            final_expand=4,
            n_classes=128,
            # act_func="gelu",
            act_func="hswish",
            **build_kwargs_from_config(kwargs, SegHead),
        )
    model = EfficientViTDepth(backbone, head)
    return model

def efficientvit_depths_l0(**kwargs) -> EfficientViTDepth:
    from rgbmodule.rgbextractor.efficientvit.backbone import efficientvit_depth_l0

    backbone = efficientvit_depth_l0(**kwargs)

    head = SegHead(
            fid_list=["stage4", "stage3", "stage2"],
            in_channel_list=[128, 64, 32],
            stride_list=[32, 16, 8],
            head_stride=8,
            head_width=32,
            head_depth=1,
            expand_ratio=4,
            middle_op="fmbconv",
            final_expand=4,
            n_classes=128,
            # act_func="gelu",
            act_func="hswish",
            **build_kwargs_from_config(kwargs, SegHead),
        )
    model = EfficientViTDepth(backbone, head)
    return model