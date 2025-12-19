# coding=utf-8
# import logging

# logging.basicConfig(level=logging.ERROR)

import math
import numpy as np
from distutils.version import LooseVersion

import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class QuantizeWeight_Ristretto(Function):
    def __init__(self, bw_params, fl_params, training=True, quantize=False, max_alpha=10.0, scale_factor=-1,winograd4x4=True):
        super(QuantizeWeight_Ristretto, self).__init__()
        self.bw_params = float(bw_params)
        self.fl_params = float(fl_params)
        self.training = training
        self.quantize = quantize

        #Ristretto clip
        self.max_alpha = max_alpha
        self.k = scale_factor
        self.winograd4x4 = winograd4x4

    def forward(self, weight):
        if self.training and self.k > 0:
            self.max_alpha = weight.abs().mean() * self.k
            weight.clamp_(min=-self.max_alpha, max=self.max_alpha)

        # quantize weight when train float and fix
        fmin = -(2 ** (self.bw_params - 1)) * (2 ** (-self.fl_params))
        fmax = (2 ** (self.bw_params - 1) - 1) * (2 ** (-self.fl_params))
        if not isinstance(fmin, float) and not isinstance(fmin, int):
            fmin = fmin.item()
            fmax = fmax.item()
        if self.winograd4x4:
            weight = (weight*4./9.).clamp(min=fmin, max=fmax)
            weight = torch.floor(0.5+weight/(2 ** (-self.fl_params))) * (2 ** (-self.fl_params))
            return weight * 9./4.
        else:
            weight = weight.clamp(min=fmin, max=fmax)
            weight = torch.floor(0.5+weight/(2 ** (-self.fl_params))) * (2 ** (-self.fl_params))
            return weight

    def backward(self, grad_output):
        return grad_output#, None

class QuantizeData_Ristretto(Function):

    def __init__(self, bw_layer, fl_layer, training=True, quantize=False):
        super(QuantizeData_Ristretto, self).__init__()
        self.bw_layer = float(bw_layer)
        self.fl_layer = float(fl_layer)
        self.training = training
        self.quantize = quantize

    def forward(self, data):
        # quantize weight when train float and fix
        fmin = -(2 ** (self.bw_layer - 1)) * (2 ** (-self.fl_layer))
        fmax = (2 ** (self.bw_layer - 1) - 1) * (2 ** (-self.fl_layer))
        if not isinstance(fmin, float) and not isinstance(fmin, int):
            fmin = fmin.item()
            fmax = fmax.item()
        data = data.clamp(min=fmin, max=fmax)
        data = torch.floor(0.5+data/(2 ** (-self.fl_layer))) * (2 ** (-self.fl_layer))
        return data

    def backward(self, grad_output):
        return grad_output#, None


class ConvRistretto2d(nn.Conv2d):

    def __init__(self,
                 cfg, 
                 in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 bw_layer_in=32, fl_layer_in=16,
                 bw_layer_out=32, fl_layer_out=16,
                 bw_params=32, fl_params=16,
                 bw_bias=32, fl_bias=16,
                 rounding=0,
                 winograd4x4=True):
        super(ConvRistretto2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                              padding, dilation, groups, bias)
        ####################################
        # initialize convolution parameters#
        ####################################
        # self.reset_parameters()
        self.param_init = False
        self.cfg = cfg
        self.rounding = rounding

        if torch.__version__ == "parrots":
            self.bw_params = torch.nn.Parameter(torch.Tensor([bw_params]))
            self.fl_params = torch.nn.Parameter(torch.Tensor([fl_params]))
            self.bw_bias = torch.nn.Parameter(torch.Tensor([bw_bias]))
            self.fl_bias = torch.nn.Parameter(torch.Tensor([fl_bias]))
            self.bw_layer_in = torch.nn.Parameter(torch.Tensor([bw_layer_in]))
            self.fl_layer_in = torch.nn.Parameter(torch.Tensor([fl_layer_in]))
            self.bw_layer_out = torch.nn.Parameter(torch.Tensor([bw_layer_out]))
            self.fl_layer_out = torch.nn.Parameter(torch.Tensor([fl_layer_out]))
        else:
            self.bw_params = torch.nn.Parameter(torch.Tensor([bw_params]), requires_grad=False)
            self.fl_params = torch.nn.Parameter(torch.Tensor([fl_params]), requires_grad=False)
            self.bw_bias = torch.nn.Parameter(torch.Tensor([bw_bias]), requires_grad=False)
            self.fl_bias = torch.nn.Parameter(torch.Tensor([fl_bias]), requires_grad=False)
            self.bw_layer_in = torch.nn.Parameter(torch.Tensor([bw_layer_in]), requires_grad=False)
            self.fl_layer_in = torch.nn.Parameter(torch.Tensor([fl_layer_in]), requires_grad=False)
            self.bw_layer_out = torch.nn.Parameter(torch.Tensor([bw_layer_out]), requires_grad=False)
            self.fl_layer_out = torch.nn.Parameter(torch.Tensor([fl_layer_out]), requires_grad=False)

        mask_weight = torch.ones((
            out_channels, in_channels // groups, kernel_size, kernel_size))
        self.register_buffer("mask_weight", mask_weight)
        self.quantize_output_flag = False
        self.layer_num = -1

        #Ristretto_clip
        self.winograd4x4 = winograd4x4 #(kernel_size != 1)
        self.scale_factor = cfg.SCALE_FACTOR
        if torch.__version__ == "parrots":
            self.max_alpha = torch.nn.Parameter(torch.FloatTensor([10.0]))
        else:
            self.max_alpha = torch.nn.Parameter(torch.FloatTensor([10.0]), requires_grad=False)


    def norm_param_init(self):
        self.bw_params.data[...] = self.bw_layer_in.data[...] = self.bw_layer_out.data[...] = 8
        self.fl_params.data[...] = self.fl_layer_in.data[...] = self.fl_layer_out.data[...] = 4

    def toggle_param_init(self, flag, bw_params=8, bw_layer=8, rounding=0):
        self.param_init = flag
        if flag:
            self.bw_params.data[...] = bw_params
            self.rounding = rounding
            #test add
            self.bw_layer_in.data[...] = self.bw_layer_out.data[...] = int(bw_layer)
            self.bw_bias.data[...] = 16

    def find_fl(self, data, bw=8):
        # klfunc = nn.KLDivLoss()
        klfunc = lambda d1,d2 : (d1-d2).abs().flatten().sum() / d1.abs().flatten().sum()
        result_kl = []
        arraylist = list(range(-10, 17, 1))
        for fl in arraylist:
            fmin = -(2 ** (float(bw) - 1)) * (2 ** (-fl))
            fmax = (2 ** (float(bw) - 1) - 1) * (2 ** (-fl))
            quant_data = data.clamp(min=fmin,max=fmax)
            quant_data = torch.round(quant_data/(2**-fl)) * (2**-fl)
            result_kl.append(klfunc(data, quant_data))
        fl_posi = result_kl.index(min(result_kl))
        return arraylist[fl_posi]

    def set_quantize_weight_param(self, bw_params, input, output, winograd4x4):
        self.bw_params.data[...] = int(bw_params)

        def find_weight_fl(weight, bw_params=8):
            klfunc = lambda d1,d2 : (d1-d2).abs().flatten().sum() / d1.abs().flatten().sum()
            #klfunc = nn.KLDivLoss()
            result_kl = []
            enum_list = [fl for fl in range(-10, 17, 1)]
            for fl in enum_list:
                fmin = -(2 ** (float(bw_params) - 1)) * (2 ** (-fl))
                fmax = (2 ** (float(bw_params) - 1) - 1) * (2 ** (-fl))
                if winograd4x4:
                    quant_weight = (weight*4./9.).clamp(min=fmin,max=fmax)
                    quant_weight = torch.round(quant_weight/(2**-fl)) * (2**-fl)*9./4.
                else:
                    quant_weight = (weight).clamp(min=fmin,max=fmax)
                    quant_weight = torch.round(quant_weight/(2**-fl)) * (2**-fl)

                quant_output = F.conv2d(input, quant_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
                result_kl.append(abs(klfunc(output, quant_output)))
            fl_posi = result_kl.index(min(result_kl))
            return enum_list[fl_posi]

        def find_max_weight_fl(weight, bw_params=8):
            if winograd4x4:
                quantize_threshold = (weight*4./9.).abs().max().detach().cpu().numpy()
            else:
                quantize_threshold = weight.abs().max().detach().cpu().numpy()
            return int(np.floor(np.log2(2 ** (float(bw_params) - 1) / quantize_threshold)))

        self.fl_params.data[...] = find_weight_fl(self.weight, bw_params)

    def set_quantize_layer_param(self, bw_layer, input, output):
        self.bw_layer_in.data[...] = self.bw_layer_out.data[...] = int(bw_layer)

        def find_fl(x, bw=8):
            quantize_threshold = x.abs().max().detach().cpu().numpy()
            return int(np.floor(np.log2(2 ** (float(bw) - 1) / quantize_threshold)))

        self.fl_layer_in.data[...] = find_fl(input, bw_layer)
        if self.quantize_output_flag:
            self.fl_layer_out.data[...] = find_fl(output, bw_layer)

    def set_quantize_bias_param(self, bw_bias, input, output):
        self.bw_bias.data[...] = int(bw_bias)

        def find_max_bias_fl(bias, bw_bias=8):
            quantize_threshold = bias.abs().max().detach().cpu().numpy()
            return int(np.floor(np.log2(2 ** (float(bw_bias) - 1) / quantize_threshold)))

        self.fl_bias.data[...] = find_max_bias_fl(self.bias, bw_bias)

    def forward(self, input):
        cfg = self.cfg         
        if self.param_init:
            output = F.conv2d(input, self.weight, self.bias, self.stride,
                              self.padding, self.dilation, self.groups)

            self.set_quantize_layer_param(self.bw_layer_in, input, output)
            self.set_quantize_weight_param(self.bw_params, input, output, winograd4x4=self.winograd4x4)
            self.set_quantize_bias_param(self.bw_bias, input, output)

            # print('layer_num:', self.layer_num)
            # print('bw_params:', float(self.bw_params), float(self.fl_params))
            # print('bw_layer_in:', float(self.bw_layer_in), float(self.fl_layer_in))
            if self.quantize_output_flag:
                # from lib.config import cfg
                if cfg.LOCAL_RANK == 0:
                    print('bw_layer_out(layer_num, bw, fl):', self.layer_num, float(self.bw_layer_out), float(self.fl_layer_out), float(self.bw_bias), float(self.fl_bias))
        else:
            if cfg.QUANT == True and self.layer_num == 0:
                quantize_input = QuantizeData_Ristretto(bw_layer=self.bw_layer_in.data, fl_layer=0, training=self.training, quantize=self.param_init)
            else:
                quantize_input = QuantizeData_Ristretto(bw_layer=self.bw_layer_in.data, fl_layer=self.fl_layer_in.data, training=self.training, quantize=self.param_init)

            quantize_weight = QuantizeWeight_Ristretto(
                bw_params=self.bw_params.data, fl_params=self.fl_params.data, 
                training=self.training, quantize=self.param_init, max_alpha=self.max_alpha.data,
                scale_factor=self.scale_factor, winograd4x4=self.winograd4x4)

            weight  = quantize_weight(self.weight)

            input = quantize_input(input)

            if self.bias is not None:
                quantize_bias = QuantizeData_Ristretto(bw_layer=self.bw_bias.data.item(),
                                                        fl_layer=self.fl_bias.data.item(), training=self.training,
                                                        quantize=self.param_init)
                bias = quantize_bias(self.bias)
            else:
                bias = None

            output = F.conv2d(input, weight, bias, self.stride,
                  self.padding, self.dilation, self.groups)

            # from lib.config import cfg
            if cfg.QUANT == True:
                if self.quantize_output_flag:
                    quantize_output = QuantizeData_Ristretto(bw_layer=8, fl_layer=self.fl_layer_out.data, training=self.training, quantize=self.param_init)
                    output = quantize_output(output)

            if cfg.QUANT_NET_DEBUG:
                def dump_feature_map(data, file_name, fl):
                    fl_data = torch.Tensor([(2**fl)]).type_as(data)
                    quant_data = (data * fl_data).type(torch.int8)
                    quant_data = quant_data.cpu().detach().numpy()
                    outfile = open(file_name, 'w')
                    hex_map = [0]*256
                    for i in range(256):
                        if i < 16:
                            hex_map[i] = '0' + hex(i)[2:]
                        else:
                            hex_map[i] = hex(i)[2:]
                    for i in range(quant_data.shape[1]):
                        for j in range(quant_data.shape[2]):
                            for k in range(quant_data.shape[3]):
                                # out_str = ''.join([hex_map[x] for x in quant_data[:, i]])
                                outfile.write(hex_map[quant_data[0,i,j,k]] + '\n')
                    outfile.close()
                if self.quantize_output_flag:
                    print('before rpn_out dump, idx:', self.layer_num, self.fl_layer_out.data, output.shape, output.sum())
                    if self.layer_num == 25 or self.layer_num == 31:
                        dump_output = output[:,::2,...]
                    elif self.layer_num == 29:
                        dump_output = output[:,3::4,...]
                    elif self.layer_num == 23:
                        select = torch.zeros(output.shape[1]).type(torch.uint8)
                        for i in range(output.shape[1]//8):
                            select[i+i*8] = 1
                            select[i+i*8+4] = 1
                        dump_output = output[:,select,...]
                    else:
                        dump_output = output
                    layer_map = {24:22, 23:23, 25:24, 30:28, 31:30, 29:29}
                    dump_feature_map(dump_output, 'rpn_out/%d_conv%d_rpn_out.txt' % (cfg.debug_image_idx, layer_map[self.layer_num]), fl=self.fl_layer_out.data)
                    print('rpn_out dump done, idx:', self.layer_num)
                elif self.layer_num == 1:
                    print('before rpn_in dump, idx:', self.layer_num, self.fl_layer_out.data, input.shape, input.sum())
                    dump_feature_map(input, 'rpn_in/%d_rpn_in.txt' % cfg.debug_image_idx, fl=self.fl_layer_out.data)
                    print('rpn_in dump done, idx:', self.layer_num)
                elif self.layer_num == 0:
                    cfg.debug_image_idx += 1
                    # vfe_out_quant_fl_layer_out = 6 #!!!! for debug (vfe_out_quant_fl_layer_out will be used in pointpillars.py as well)
                    # quantize_output = QuantizeData_Ristretto(bw_layer=8, fl_layer=vfe_out_quant_fl_layer_out, 
                    #     training=self.training, quantize=self.param_init)
                    # output = quantize_output(output)
                print('cfg.debug_image_idx:', cfg.debug_image_idx)
        return output


    def set_layer_num(self, layer_num):
        self.layer_num = layer_num
        cfg = self.cfg
        if cfg.QUANT == True and self.layer_num in [23, 24, 25, 29, 30, 31]:
            self.quantize_output_flag = True
            if self.layer_num == 23:
                self.fl_layer_out.data[...] = 5
            elif self.layer_num == 24: # box
                self.fl_layer_out.data[...] = 6
            elif self.layer_num == 25:
                self.fl_layer_out.data[...] = 5
            elif self.layer_num == 29:
                self.fl_layer_out.data[...] = 5
            if self.layer_num == 30: # box
                self.fl_layer_out.data[...] = 6
            elif self.layer_num == 31:
                self.fl_layer_out.data[...] = 5

class SparseConvRistretto2d(nn.Module):

    def __init__(self,
                 cfg, 
                 in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 bw_layer_in=32, fl_layer_in=16,
                 bw_layer_out=32, fl_layer_out=16,
                 bw_params=32, fl_params=16,
                 bw_bias=32, fl_bias=16,
                 rounding=0, winograd4x4=True):
        super().__init__()
        self.conv = ConvRistretto2d(cfg, in_channels, out_channels, kernel_size, stride,
                                              padding, dilation, groups, bias, winograd4x4=winograd4x4)

    def forward(self, x, sparse_masks=None):
        output = self.conv(x)
        if sparse_masks is not None:
            flag = True
            for mask in sparse_masks:
                if output.shape[2:] == mask.shape[2:]:
                    output = output * mask.repeat(1,output.shape[1],1,1)
                    # output[mask.repeat(1,output.shape[1],1,1)==0]=-100000
                    flag = False
                    break
            if flag:
                print(x.shape,'! sparse_masks shape is wrong!')
        return output


def set_layer_num(model):
    modules = model.modules()
    idx = 0
    # for name, module in model._modules.items():
    #     print("名称:{}".format(name))
    for module in modules:
        if isinstance(module, ConvRistretto2d):
            print(idx, module)
            module.set_layer_num(idx)
            idx += 1
    return model

def set_fix_point_param(model, fake_input=None):
    model = set_layer_num(model)
    print('init net model para:')
    if fake_input == None:
        import pickle
        fake_input = pickle.load(open('quant_input.pkl', 'rb'))
    model.eval()

    for module in model.modules():
        if isinstance(module, ConvRistretto2d):
            module.toggle_param_init(True, bw_params=8, bw_layer=8)  # bw_layer_in = bw_layer_out = bw_layer

    with torch.set_grad_enabled(False):
        model.forward(fake_input)  # forward the whole net once and quantization parameters will be automatically set.
    del fake_input

    for module in model.modules():
        if isinstance(module, ConvRistretto2d):
            module.toggle_param_init(False)  # disable the parameters search.

    # check model
    ln = 0
    for module in model.modules():
        if isinstance(module, ConvRistretto2d):
            print('layer is', module, 'param is', module.bw_layer_in, module.fl_layer_in, module.bw_params, module.fl_params)
    return model
