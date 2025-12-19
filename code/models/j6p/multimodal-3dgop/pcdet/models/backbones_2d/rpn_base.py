import sys
import torch
import torch.nn as nn
from collections import OrderedDict
from functools import partial
from pcdet.utils.common_utils import save_np
import os

class GroupNorm(nn.GroupNorm):
    def __init__(self, num_channels, num_groups, eps=1e-5, affine=True):
        super().__init__(num_groups=num_groups, num_channels=num_channels, eps=eps, affine=affine)


class Empty(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Empty, self).__init__()

    def forward(self, *args, **kwargs):
        if len(args) == 1:
            return args[0]
        elif len(args) == 0:
            return None
        return args


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



class RPNBase(nn.Module):
    """
    RPN components without head.
    """
    def __init__(self,
                 model_cfg, input_channels
                 ):
        super(RPNBase, self).__init__()
        self.model_cfg = model_cfg
        use_norm = self.model_cfg.USE_NORM
        layer_nums = self.model_cfg.LAYER_NUMS
        layer_strides = self.model_cfg.LAYER_STRIDES
        num_filters = self.model_cfg.NUM_FILTERS
        upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
        num_input_features = self.model_cfg.NUM_INPUT_FEATURES
        input_channels = num_input_features
        use_groupnorm = self.model_cfg.USE_GROUPNORM
        num_groups = self.model_cfg.NUM_GROUPS
        use_deconv = self.model_cfg.USE_DECONV
        pre_conv = self.model_cfg.PRE_CONV
        assert len(layer_strides) == len(layer_nums)
        assert len(num_filters) == len(layer_nums)
        assert len(upsample_strides) == len(layer_nums)
        assert len(num_upsample_filters) == len(layer_nums)
        """
        factors = []
        for i in range(len(layer_nums)):
            assert int(np.prod(layer_strides[:i + 1])) % upsample_strides[i] == 0
            factors.append(np.prod(layer_strides[:i + 1]) // upsample_strides[i])
        assert all([x == factors[0] for x in factors])
        """
        if use_norm:
            if use_groupnorm:
                BatchNorm2d = partial(GroupNorm, num_groups=num_groups, eps=1e-3)
            else:
                BatchNorm2d = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
            Conv2d = partial(nn.Conv2d, bias=False)
            ConvTranspose2d = partial(nn.ConvTranspose2d, bias=False)
        else:
            BatchNorm2d = Empty
            Conv2d = partial(nn.Conv2d, bias=True)
            ConvTranspose2d = partial(nn.ConvTranspose2d, bias=True)

        in_filters = [num_input_features, *num_filters[:-1]]
        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.
        blocks = []
        deblocks = []
        fpn_blocks = []
        for i, layer_num in enumerate(layer_nums):
            groups = 1
            block = Sequential(
                #nn.ZeroPad2d(1),
                Conv2d(in_filters[i], num_filters[i], 3, stride=layer_strides[i], padding=1, groups=groups),
                BatchNorm2d(num_filters[i]),
                nn.ReLU(inplace=True),
            )
            
            for j in range(layer_num):
                groups = self.model_cfg.CONVGROUPS[i] if j != layer_num-1 else 1
                block.add(Conv2d(num_filters[i], num_filters[i], 3, padding=1, groups=groups))
                block.add(BatchNorm2d(num_filters[i]))
                block.add(nn.ReLU(inplace=True))
            blocks.append(block)

            if self.model_cfg.PP_FPN_HEAVY:
                if use_deconv:
                    deconv = ConvTranspose2d(num_filters[i], num_upsample_filters[i], upsample_strides[i],
                                        stride=upsample_strides[i])
                    upsample = Empty()
                else:
                    deconv = Conv2d(num_filters[i], num_upsample_filters[i], kernel_size=3, padding=1)
                    upsample = nn.Upsample(scale_factor=upsample_strides[i])
                if pre_conv and not use_deconv:
                    deblock = Sequential(
                        deconv,
                        BatchNorm2d(num_upsample_filters[i]),
                        nn.ReLU(inplace=True),
                        upsample
                    )
                if not pre_conv and not use_deconv:
                    deblock = Sequential(
                        upsample,
                        deconv,
                        BatchNorm2d(num_upsample_filters[i]),
                        nn.ReLU(inplace=True)
                    )
                if use_deconv:
                    if i == 0:
                        deblock = Sequential(
                            Conv2d(num_filters[i], num_upsample_filters[i], kernel_size=3, padding=1),
                            BatchNorm2d(num_upsample_filters[i]),
                            nn.ReLU(inplace=True)
                        )
                    elif i == 1:
                        deblock = Sequential(
                            Conv2d(num_filters[i], num_upsample_filters[i], kernel_size=3, padding=1),
                            BatchNorm2d(num_upsample_filters[i]),
                            nn.ReLU(inplace=True),
                            ConvTranspose2d(num_upsample_filters[i], num_upsample_filters[i], 2, stride=2)
                        )
                    else:
                        deblock = Sequential(
                            Conv2d(num_filters[i], num_upsample_filters[i], kernel_size=3, padding=1),
                            BatchNorm2d(num_upsample_filters[i]),
                            nn.ReLU(inplace=True),
                            ConvTranspose2d(num_upsample_filters[i], num_upsample_filters[i], 2, stride=2),
                            Conv2d(num_upsample_filters[i], num_upsample_filters[i], kernel_size=3, padding=1),
                            BatchNorm2d(num_upsample_filters[i]),
                            nn.ReLU(inplace=True),
                            ConvTranspose2d(num_upsample_filters[i], num_upsample_filters[i], 2, stride=2)
                        )
                deblocks.append(deblock)

        self.blocks = nn.ModuleList(blocks)
        if self.model_cfg.PP_FPN_HEAVY:
            self.deblocks = nn.ModuleList(deblocks)
            if self.model_cfg.FPN_SUM:
                self.fpn_merge = Conv2d(num_upsample_filters[0], num_upsample_filters[0], kernel_size=1, padding=0)
                # self.fpn_merge = Conv2d(num_upsample_filters[0], num_upsample_filters[0], kernel_size=3, padding=1)
        
        self.num_bev_features = num_upsample_filters[0]

    def forward(self, data_dict):
        x = data_dict['spatial_features']
        ups_stride2 = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            features = self.deblocks[i](x)
            ups_stride2.append(features)
        if not self.model_cfg.FPN_SUM:
            ups_stride2_final = torch.cat(ups_stride2, dim=1)
        else:
            #ups_stride2_final = ups_stride2[0] + ups_stride2[1] + ups_stride2[2]
            y = ups_stride2[1] + ups_stride2[2]
            ups_stride2_final = ups_stride2[0] + self.fpn_merge(y)
        
        data_dict['spatial_features_2d_lidar'] = ups_stride2_final
        if os.getenv("CALIB") == 'True':
            calib_path = os.getenv("CALIB_PATH")
            save_dir = os.path.join(calib_path, 'lidar-bev-backbone')
            save_np(os.path.join(save_dir, "outputs/spatial_features_2d_lidar/{}".format(self.cnt)), data_dict['spatial_features_2d_lidar'])
            save_dir = os.path.join(calib_path, 'lidar-branch')
            save_np(os.path.join(save_dir, "outputs/spatial_features_2d_lidar/{}".format(self.cnt)), data_dict['spatial_features_2d_lidar'])
        return data_dict

if __name__ == '__main__':
    from easydict import EasyDict as edict
    model_cfg = edict()
    model_cfg.USE_NORM = True
    model_cfg.LAYER_NUMS = [4, 5, 6]
    model_cfg.LAYER_STRIDES = [2, 2, 2]
    model_cfg.NUM_FILTERS = [128, 160, 192]
    model_cfg.UPSAMPLE_STRIDES = [1, 2, 4]
    model_cfg.NUM_UPSAMPLE_FILTERS = [96, 96, 96]
    model_cfg.NUM_INPUT_FEATURES = 32
    model_cfg.USE_GROUPNORM = False
    model_cfg.NUM_GROUPS = 16
    model_cfg.USE_DECONV = True
    model_cfg.PRE_CONV = True
    model_cfg.CONVGROUPS = [1, 1, 1]
    model_cfg.PP_FPN_HEAVY = True
    model_cfg.FPN_SUM = True
    print(model_cfg)
    model = RPNBase(model_cfg)
    print(model)
    