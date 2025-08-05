# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os
import torch
import torch.nn as nn
import torch.types

from torchvision import transforms

from nav_tasks import NAVSUITE_TASKS_DATA_DIR, mdp


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class PerceptNet(nn.Module):
    """
    A convolutional neural network for perceptual feature extraction, based on a customizable ResNet-like architecture.

    Args:
        layers (list[int]): A list specifying the number of blocks for each layer of the network.
        block (nn.Module, optional): The building block class to use within each layer, such as `BasicBlock`.
            Default is `BasicBlock`.
        groups (int, optional): The number of groups for grouped convolutions. Default is 1.
        width_per_group (int, optional): The width of each group. Default is 64.
        replace_stride_with_dilation (list[bool], optional): A list of booleans indicating whether to replace strides
            with dilated convolutions in layers 2, 3, and 4. Default is [False, False, False].
        norm_layer (nn.Module, optional): The normalization layer to use (e.g., `nn.BatchNorm2d`).
            Default is `nn.BatchNorm2d`.

    Attributes:
        conv1 (nn.Conv2d): The initial convolutional layer with a kernel size of 7x7.
        relu (nn.ReLU): ReLU activation function applied after the first convolution.
        maxpool (nn.MaxPool2d): Max pooling layer with a kernel size of 3x3 and a stride of 2.
        layer1, layer2, layer3 (nn.Sequential): Sequential layers consisting of blocks defined by the `block`
            argument.
        global_avg_pool (nn.AdaptiveAvgPool2d): Global average pooling layer that outputs a tensor of size 1x1 per
            channel.

    Methods:
        _make_layer(block, planes, blocks, stride=1, dilate=False): Helper function to construct layers with the
            specified number of blocks, channels, and dilation.
        _forward_impl(x): Implements the forward pass of the network.
        forward(x): Defines the standard forward pass interface for the network.

    Example Usage:
        >>> model = PerceptNet([2, 2, 2, 2], block=BasicBlock)
        >>> output = model(torch.randn(1, 3, 360, 640))
        >>> print(output.shape)  # Expected output shape: [1, 256, 1, 1]
    """

    def __init__(
        self, layers, block=BasicBlock, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None
    ):

        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None or a 3-element tuple, got {}".format(
                    replace_stride_with_dilation
                )
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # fixed layer, average pools each channel to 1x1

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):  # x: [N, 3, 360, 640]
        x = self.conv1(x)  # x_new: [N, 64, 180, 320]
        x = self.relu(x)  # x_new: [N, 64, 180, 320]
        x = self.maxpool(x)  # x_new: [N, 64, 90, 160]

        x = self.layer1(x)  # x_new: [N, 64, 90, 160]
        x = self.layer2(x)  # x_new: [N, 128, 45, 80]
        x = self.layer3(x)  # x_new: [N, 256, 23, 40]
        # Note: layer4 from the pretrained model isn't used.

        x = self.global_avg_pool(x)  # x_new: [N, 256, 1,1]
        return x

    def forward(self, x):
        return self._forward_impl(x)


class DepthEmbedder:
    device: str
    encoder_net: torch.nn.Module

    def __init__(self, device: str):
        # Initialize device and load the model
        self.device = device
        self.encoder_net = mdp.PerceptNet(layers=[2, 2, 2, 2]).to(self.device)
        resnet = os.path.join(NAVSUITE_TASKS_DATA_DIR, "Policies", "perceptnet_emb256_low_resolution_SD.pt")
        self.encoder_net.load_state_dict(torch.load(resnet))
        self.encoder_net.eval()

    @staticmethod
    def _preprocess_image(image_tensor) -> torch.Tensor:
        """
        Preprocesses a grayscale depth image tensor by copying it to 3 channels and resizing.

        Args:
            image_tensor (torch.Tensor): A tensor of shape (envs x H x W) representing the grayscale depth image.

        Returns:
            torch.Tensor: A tensor of shape (envs x 3 x H x W) with the image copied to 3 channels and resized to (180, 320).
        """
        image_tensor = image_tensor.unsqueeze(1)
        image_tensor = image_tensor.repeat(1, 3, 1, 1)
        preprocess = transforms.Resize((180, 320), antialias=True)
        return preprocess(image_tensor)

    def process_image(self, image_tensor) -> torch.Tensor:
        """
        Processes a depth image tensor by embedding it using the PerceptNet.

        Args:
            image_tensor (torch.Tensor): A tensor of shape (envs x H x W) representing the grayscale depth image.

        Returns:
            torch.Tensor: A tensor of shape (envs x 256) representing the embedded depth image
        """
        with torch.no_grad():
            preprocessed_image = self._preprocess_image(image_tensor).to(self.device)
            output = self.encoder_net(preprocessed_image)
            return output.flatten(1, -1)


class DepthEmbedderSingleton:
    """A global singleton for the DepthEmbedder."""

    _embedder_instance = None

    @classmethod
    def get_embedder(cls, device):
        if cls._embedder_instance is None:
            cls._embedder_instance = DepthEmbedder(device)
        return cls._embedder_instance
