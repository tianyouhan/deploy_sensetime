import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class Sequential(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        for name, module in kwargs.items():
            if sys.version_info < (3, 6):
                raise ValueError("kwargs only supported in py36+")
            if name in self._modules:
                raise ValueError("name exists.")
            self.add_module(name, module)

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def add(self, module, name=None):
        if name is None:
            name = str(len(self._modules))
            if name in self._modules:
                raise KeyError("name exists")
        self.add_module(name, module)

    def forward(self, input, sparse_masks=None):
        for module in self._modules.values():
            input = module(input)
        return input


class Empty(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Empty, self).__init__()

    def forward(self, *args, **kwargs):
        if len(args) == 1:
            return args[0]
        elif len(args) == 0:
            return None
        return args


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, droprate=0.0):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
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
        self.droprate = droprate

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck_V15(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, droprate=0.0):
        super(Bottleneck_V15, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # width = int(planes/self.expansion)
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

    def forward(self, x):
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


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, droprate=0.0):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # width = int(planes/self.expansion)
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, stride)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride=1, groups=groups, dilation=dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
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


class RPNBase_ResNet(nn.Module):
    """
    RPN components without head.
    """
    def __init__(self,
                 resblock='BasicBlock',
                 use_deconv=False,
                 pre_conv=True,
                 num_input_features=64,
                 num_upsample_filters=(64, 64, 64),
                 upsample_strides=(1, 2, 4),
                 num_res_stages=3,
                 num_res_blocks=(5, 15, 16),
                 res_stage_dilation=(False, False, False),
                 num_res_filters=(128, 160, 192),
                 res_stage_1_split_conv=True,
                 groups=1,
                 width_per_group=64,
                 norm_layer=None,
                 zero_init_residual=False,
                 droprate=0.0,
                 **kwargs):
        super(RPNBase_ResNet, self).__init__()
        assert num_res_stages <= 5 and num_res_stages >=3
        assert len(upsample_strides) == num_res_stages
        assert len(num_res_blocks) == num_res_stages
        assert len(num_res_filters) == num_res_stages
        assert len(res_stage_dilation) == num_res_stages
        assert len(num_upsample_filters) == num_res_stages
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = num_input_features
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.droprate = droprate

        blocks = []
        deblocks = []
        if resblock == 'BasicBlock':
            block = BasicBlock
        elif resblock == 'Bottleneck':
            block = Bottleneck
        elif resblock == 'Bottleneck_V15':
            block = Bottleneck_V15
        else:
            raise ValueError("resblock not available")

        if res_stage_1_split_conv:
            stage_1 = Sequential(conv3x3(self.inplanes, num_res_filters[0], stride=2),
                norm_layer(num_res_filters[0]),
                nn.ReLU(inplace=True))
            for i in range(num_res_blocks[0]-1):
                stage_1.add(conv3x3(num_res_filters[0], num_res_filters[0], stride=1))
                stage_1.add(norm_layer(num_res_filters[0]))
                stage_1.add(nn.ReLU(inplace=True))
        else:
            stage_1 = Sequential(nn.Conv2d(self.inplanes, num_res_filters[0], kernel_size=7, stride=2, padding=3, bias=False),
                norm_layer(num_res_filters[0]),
                nn.ReLU(inplace=True))
        blocks.append(stage_1)
        self.inplanes = num_res_filters[0]

        stage_2 = Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        stage_2.add(self._make_layer(block, num_res_filters[1], num_res_blocks[1], stride=1, dilate=False))
        blocks.append(stage_2)

        stage_3 = self._make_layer(block, num_res_filters[2], num_res_blocks[2], stride=2, dilate=res_stage_dilation[2])
        blocks.append(stage_3)

        if num_res_stages >= 4:
            stage_4 = self._make_layer(block, num_res_filters[3], num_res_blocks[3], stride=2, dilate=res_stage_dilation[3])
            blocks.append(stage_4)

        if num_res_stages == 5:
            stage_5 = self._make_layer(block, num_res_filters[4], num_res_blocks[4], stride=2, dilate=res_stage_dilation[4])
            blocks.append(stage_5)

        for i in range(num_res_stages):
            if i == 0:
                de_input_ch = num_res_filters[0]
            else:
                de_input_ch = num_res_filters[i] * block.expansion
            if use_deconv:
                deconv = nn.ConvTranspose2d(de_input_ch, num_upsample_filters[i], upsample_strides[i],
                                    stride=upsample_strides[i], bias=False)
                upsample = Empty()
            else:
                deconv = nn.Conv2d(de_input_ch, num_upsample_filters[i], kernel_size=3, padding=1, bias=False)
                upsample = nn.Upsample(scale_factor=upsample_strides[i])
            if pre_conv or use_deconv:
                deblock = Sequential(
                    deconv,
                    norm_layer(num_upsample_filters[i]),
                    nn.ReLU(inplace=True),
                    upsample
                )
            else:
                deblock = Sequential(
                    upsample,
                    deconv,
                    norm_layer(num_upsample_filters[i]),
                    nn.ReLU(inplace=True)
                )
            deblocks.append(deblock)

        self.blocks = nn.ModuleList(blocks)
        self.deblocks = nn.ModuleList(deblocks)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, Bottleneck_V15):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, droprate=self.droprate))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, droprate=self.droprate))

        return Sequential(*layers)

    def forward(self, x, sparse_masks=None):
        ups = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            ups.append(self.deblocks[i](x))
        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        else:
            x = ups[0]
        return x