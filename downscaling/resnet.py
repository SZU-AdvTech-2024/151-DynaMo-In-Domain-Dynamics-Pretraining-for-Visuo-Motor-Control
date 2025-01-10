import torch
import torchvision
import torch.nn as nn


class resnet18(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        output_dim: int = 512,  # fixed for resnet18; included for consistency with config
        unit_norm: bool = False,
    ):
        super().__init__()
        resnet = torchvision.models.resnet18(pretrained=pretrained)
        # 修改第一层卷积的输入通道数
        resnet.conv1 = nn.Conv2d(
            in_channels=54,  # 修改为54通道
            out_channels=resnet.conv1.out_channels,  # 保持输出通道数不变
            kernel_size=resnet.conv1.kernel_size,
            stride=resnet.conv1.stride,
            padding=resnet.conv1.padding,
            bias=resnet.conv1.bias,
        )
        # 如果使用预训练权重，初始化新通道的权重
        if pretrained:
            old_weights = resnet.conv1.weight.clone()
            new_weights = torch.zeros((6, 54, 32, 64))  # 新的权重张量
            new_weights[:, :3, :, :] = old_weights  # 保留原有3通道权重
            new_weights[:, 3:, :, :] = old_weights[:, :2, :, :]  # 初始化新增通道
            resnet.conv1.weight = nn.Parameter(new_weights)

        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.flatten = nn.Flatten()
        self.pretrained = pretrained
        self.normalize = torchvision.transforms.Normalize(
            mean = [0.5] * 54,  std = [0.2]*54
            # mean=[0.485, 0.456, 0.406, 0.5, 0.5], std=[0.229, 0.224, 0.225, 0.1, 0.1]
        )
        self.unit_norm = unit_norm

    def forward(self, x):
        dims = len(x.shape)
        orig_shape = x.shape
        if dims == 3:
            x = x.unsqueeze(0)
        elif dims > 4:
            # flatten all dimensions to batch, then reshape back at the end
            x = x.reshape(-1, *orig_shape[-3:])
        x = self.normalize(x)
        out = self.resnet(x)
        out = self.flatten(out)
        if self.unit_norm:
            out = torch.nn.functional.normalize(out, p=2, dim=-1)
        if dims == 3:
            out = out.squeeze(0)
        elif dims > 4:
            out = out.reshape(*orig_shape[:-3], -1)
        return out
