import torchvision.models
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor


class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride, padding=padding, dilation=dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x



def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
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
    # return SeparableConv2d(in_planes, out_planes, 3, stride, padding=dilation, bias=False)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    # return SeparableConv2d(in_planes, out_planes, 1, stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        dim_pred: int = 1,
        in_ch=512,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
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
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group

        self.convs = nn.Sequential(
            nn.Conv2d(in_ch, 512, kernel_size=3, stride=2, padding=1, bias=True),
            norm_layer(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=True),
            norm_layer(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=True),
            norm_layer(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, self.inplanes, kernel_size=3, stride=1, padding=1, bias=True),
            norm_layer(self.inplanes),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # self.fc_md = nn.Linear(512 * block.expansion, dim_pred)
        
        self.emb_st = nn.Linear(1, 64)
        self.emb_eye = nn.Linear(1, 64)
        self.emb_age = nn.Linear(1, 64)
        self.fc_md = nn.Linear(512 * block.expansion+64*2, 1)
        self.fc_se = nn.Linear(512 * block.expansion+64*2, 1024)
        self.fc_pd = nn.Linear(512 * block.expansion+64*2, 1024)

        self.map_se = nn.Sequential(
            nn.Conv2d(1024, 512, 3, 1, 1),
            nn.UpsamplingBilinear2d(scale_factor=3.0),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.UpsamplingBilinear2d(scale_factor=3.0),
            nn.Conv2d(512, 1, 3, 1, 1),
            # nn.Sigmoid(),
            
        )
        self.map_pd = nn.Sequential(
            nn.Conv2d(1024, 512, 3, 1, 1),
            nn.UpsamplingBilinear2d(scale_factor=3.0),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.UpsamplingBilinear2d(scale_factor=3.0),
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.Conv2d(256, 1, 1, 1, 0),
            # nn.Sigmoid(),
        )

        self.sig = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
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

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.convs(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc_md(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
  
    def forward_md(self, x, st, eye, age):
        x = self.convs(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x_st = self.emb_st(st)
        # x_eye = self.emb_eye(eye)
        # x_age = self.emb_age(age)
        f_comb = torch.concatenate([x, x_st], dim=1)
        out = self.fc_md(f_comb)
        return out

    def forward_se(self, x, st, eye, age):
        x = self.convs(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x_st = self.emb_st(st)
        # x_eye = self.emb_eye(eye)
        # x_age = self.emb_age(age)
        f_comb = torch.concatenate([x, x_st], dim=1)
        feat = self.fc_se(f_comb)
        # b*512
        feat = feat.view(-1, 512, 1, 1)
        out = self.map_se(feat)
        return out
  
    def forward_pd(self, x, st, eye, age):
        x = self.convs(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x_st = self.emb_st(st)
        # x_eye = self.emb_eye(eye)
        # x_age = self.emb_age(age)
        f_comb = torch.concatenate([x, x_st], dim=1)
        feat = self.fc_pd(f_comb)
        feat = feat.view(-1, 512, 1, 1)
        out = self.map_pd(feat)

        return out


    def forward_mt(self, x, st, age):
        x = self.convs(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x_st = self.emb_st(st)
        x_age = self.emb_age(age)
        f_comb = torch.concatenate([x, x_st, x_age], dim=1)

        # MD
        out_md = -self.fc_md(f_comb)
        # SE
        feat_se = self.fc_se(f_comb)
        feat_se = feat_se.view(-1, 1024, 1, 1)
        out_se = self.map_se(feat_se)
        # PD
        feat_pd = self.fc_pd(f_comb)
        feat_pd = feat_pd.view(-1, 1024, 1, 1)
        out_pd = self.map_pd(feat_pd)

        return out_md, out_se, out_pd

def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    dim_pred=1,
    in_ch=230,
    **kwargs: Any,
) -> ResNet:

    model = ResNet(block, layers, dim_pred=dim_pred, in_ch=in_ch,  **kwargs)

    return model


def resnet18(dim_pred=1, in_ch=230) -> ResNet:
    return _resnet(BasicBlock, [2, 2, 2, 2], dim_pred=dim_pred, in_ch=in_ch)

def resnet34(dim_pred=1, in_ch=230) -> ResNet:
    return _resnet(BasicBlock, [3, 4, 6, 3], dim_pred=dim_pred, in_ch=in_ch)

def resnet50(dim_pred=1, in_ch=500) -> ResNet:
    return _resnet(Bottleneck, [3, 4, 6, 3], dim_pred=dim_pred, in_ch=in_ch)

def resnet101(dim_pred=1, in_ch=500):
    return _resnet(Bottleneck, [3, 4, 23, 3], dim_pred=dim_pred, in_ch=in_ch)
