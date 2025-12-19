import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
from torch import Tensor
from torch.jit.annotations import List
import sys

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


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False, dense_layer_type='N'):
        super(_DenseLayer, self).__init__()
        self.dense_layer_type = dense_layer_type
        if self.dense_layer_type == 'B':
            self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
            self.add_module('relu1', nn.ReLU(inplace=True)),
            self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                               growth_rate, kernel_size=1, stride=1,
                                               bias=False)),
            self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
            self.add_module('relu2', nn.ReLU(inplace=True)),
            self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                               kernel_size=3, stride=1, padding=1,
                                               bias=False)),
        elif self.dense_layer_type == 'N':
            self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
            self.add_module('relu1', nn.ReLU(inplace=True)),
            self.add_module('conv1', nn.Conv2d(num_input_features, growth_rate,
                                               kernel_size=3, stride=1, padding=1,
                                               bias=False)),
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs):
        # type: (List[Tensor]) -> Tensor
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, input):
        # type: (List[Tensor]) -> bool
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    #@torch.jit.unused  # noqa: T484
    def call_checkpoint_bottleneck(self, input):
        # type: (List[Tensor]) -> Tensor
        def closure(*inputs):
            return self.bn_function(inputs)

        return cp.checkpoint(closure, *input)

    # @torch.jit._overload_method  # noqa: F811
    # def forward(self, input):
    #     # type: (List[Tensor]) -> (Tensor)
    #     pass

    # @torch.jit._overload_method  # noqa: F811
    # def forward(self, input):
    #     # type: (Tensor) -> (Tensor)
    #     pass

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input):  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            # if torch.jit.is_scripting():
            #     raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        if self.dense_layer_type == 'B':
            new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        elif self.dense_layer_type == 'N':
            new_features = bottleneck_output

        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features


class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False, dense_layer_type='N'):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
                dense_layer_type=dense_layer_type
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class RPNBase_DenseNet(nn.Module):
    """
    RPN components without head.
    """
    def __init__(self,
                 dense_layer_type='N',
                 use_deconv=False,
                 pre_conv=True,
                 num_input_features=64,
                 num_upsample_filters=(64, 64, 64),
                 upsample_strides=(1, 2, 4),
                 num_dense_stages=3,
                 num_dense_blocks=(5, 15, 16),
                 num_dense_filters=(128, 32, 32),
                 droprate=0.0,
                 bn_size=4,
                 memory_efficient=False,
                 **kwargs):
        super(RPNBase_DenseNet, self).__init__()
        assert num_dense_stages <= 5 and num_dense_stages >=3
        assert len(upsample_strides) == num_dense_stages
        assert len(num_dense_blocks) == num_dense_stages
        assert len(num_dense_filters) == num_dense_stages
        assert len(num_upsample_filters) == num_dense_stages

        blocks = []
        deblocks = []
        blocks_features = []

        stage_1 = Sequential(conv3x3(num_input_features, num_dense_filters[0], stride=2),
                nn.BatchNorm2d(num_dense_filters[0]),
                nn.ReLU(inplace=True))
        for i in range(num_dense_blocks[0]-1):
            stage_1.add(conv3x3(num_dense_filters[0], num_dense_filters[0], stride=1))
            stage_1.add(nn.BatchNorm2d(num_dense_filters[0]))
            stage_1.add(nn.ReLU(inplace=True))
        blocks.append(stage_1)
        num_features = num_dense_filters[0]
        blocks_features.append(num_features)

        stage_2 = Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        dense_block = _DenseBlock(
                num_layers=num_dense_blocks[1],
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=num_dense_filters[1],
                drop_rate=droprate,
                memory_efficient=memory_efficient,
                dense_layer_type=dense_layer_type
            )
        stage_2.add(dense_block)
        blocks.append(stage_2)
        num_features = num_features + num_dense_blocks[1] * num_dense_filters[1]
        blocks_features.append(num_features)

        for i in range(2,num_dense_stages):
            trans = _Transition(num_input_features=num_features,
                num_output_features=num_features // 2)
            net_stage = Sequential(trans)
            num_features = num_features // 2
            dense_block = _DenseBlock(
                num_layers=num_dense_blocks[i],
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=num_dense_filters[i],
                drop_rate=droprate,
                memory_efficient=memory_efficient,
                dense_layer_type=dense_layer_type
            )
            net_stage.add(dense_block)
            blocks.append(net_stage)
            num_features = num_features + num_dense_blocks[i] * num_dense_filters[i]
            blocks_features.append(num_features)

        for i in range(num_dense_stages):
            if use_deconv:
                deconv = nn.ConvTranspose2d(blocks_features[i], num_upsample_filters[i], upsample_strides[i],
                                    stride=upsample_strides[i], bias=False)
                upsample = Empty()
            else:
                deconv = nn.Conv2d(blocks_features[i], num_upsample_filters[i], kernel_size=3, padding=1, bias=False)
                upsample = nn.Upsample(scale_factor=upsample_strides[i])
            if pre_conv or use_deconv:
                if i == 0:
                    deblock = Sequential(
                        deconv,
                        nn.BatchNorm2d(num_upsample_filters[i]),
                        nn.ReLU(inplace=True),
                        upsample
                    )
                else:
                    deblock = Sequential(
                        nn.BatchNorm2d(blocks_features[i]),
                        nn.ReLU(inplace=True),
                        deconv,
                        nn.BatchNorm2d(num_upsample_filters[i]),
                        nn.ReLU(inplace=True),
                        upsample
                    )
            else:
                if i == 0:
                    deblock = Sequential(
                        upsample,
                        deconv,
                        nn.BatchNorm2d(num_upsample_filters[i]),
                        nn.ReLU(inplace=True)
                    )
                else:
                    deblock = Sequential(
                        nn.BatchNorm2d(blocks_features[i]),
                        nn.ReLU(inplace=True),
                        upsample,
                        deconv,
                        nn.BatchNorm2d(num_upsample_filters[i]),
                        nn.ReLU(inplace=True)
                    )
            deblocks.append(deblock)

        self.blocks = nn.ModuleList(blocks)
        self.deblocks = nn.ModuleList(deblocks)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

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