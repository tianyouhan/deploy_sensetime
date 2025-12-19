import sys
import torch
import torch.nn as nn
from collections import OrderedDict
from functools import partial
import math
from .Ristretto import ConvRistretto2d

class Sequential(torch.nn.Module):
    r"""A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.

    To make it easier to understand, given is a small example::

        # Example of using Sequential
        model = Sequential(
                  nn.Conv2d(1,20,5),
                  nn.ReLU(),
                  nn.Conv2d(20,64,5),
                  nn.ReLU()
                )

        # Example of using Sequential with OrderedDict
        model = Sequential(OrderedDict([
                  ('conv1', nn.Conv2d(1,20,5)),
                  ('relu1', nn.ReLU()),
                  ('conv2', nn.Conv2d(20,64,5)),
                  ('relu2', nn.ReLU())
                ]))

        # Example of using Sequential with kwargs(python 3.6+)
        model = Sequential(
                  conv1=nn.Conv2d(1,20,5),
                  relu1=nn.ReLU(),
                  conv2=nn.Conv2d(20,64,5),
                  relu2=nn.ReLU()
                )
    """

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

    def forward(self, input):
        # i = 0
        for module in self._modules.values():
            # print(i)
            input = module(input)
            # i += 1
        return input

class MaskNet(nn.Module):
    def __init__(self, expend_num=1):
        super().__init__()
        self.expend_num = expend_num
        if self.expend_num > 0:
            self.relu = nn.ReLU(inplace=True)
            self.negate_plus = ConvRistretto2d(1, 1, kernel_size=3, stride=2, padding=1, bias=True)

            self.net_module = Sequential()
            for i in range(expend_num-1):
                self.net_module.add(nn.ConstantPad2d(1, 1))
                self.net_module.add(ConvRistretto2d(1, 1, kernel_size=3, stride=1, padding=0, bias=True))
                self.net_module.add(nn.ReLU(inplace=True))

            self.negate0 = ConvRistretto2d(1, 1, kernel_size=1, stride=1, bias=True)
            self.negate1 = ConvRistretto2d(1, 1, kernel_size=1, stride=2, bias=True)
            self.negate2 = ConvRistretto2d(1, 1, kernel_size=1, stride=4, bias=True)

            self.param_init()

    def param_init(self):
        self.negate_plus.weight.data.fill_(-9)
        self.negate_plus.bias.data.fill_(1)

        for module in self.net_module:
            if isinstance(module, ConvRistretto2d):
                module.weight.data.fill_(9)
                module.bias.data.fill_(-80)

        self.negate0.weight.data.fill_(-9)
        self.negate0.bias.data.fill_(1)
        self.negate1.weight.data.fill_(-9)
        self.negate1.bias.data.fill_(1)
        self.negate2.weight.data.fill_(-9)
        self.negate2.bias.data.fill_(1)

        for module in self.modules():
            if isinstance(module, ConvRistretto2d):
               module.norm_param_init()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, mask):
        if self.expend_num <= 0:
            return None
        # print('\n0:', float(mask.sum())/float(mask.numel()))
        mask = self.relu(self.negate_plus(mask))
        # print('\n1:', 1.-float(mask.sum())/float(mask.numel()))
        mask = self.net_module(mask)
        # print('\nx:', 1.-float(mask.sum())/float(mask.numel()))
        mask0 = self.relu(self.negate0(mask))
        mask1 = self.relu(self.negate1(mask))
        mask2 = self.relu(self.negate2(mask))
        return mask0, mask1, mask2



if __name__ == '__main__':
    pass
