import torch
import torch.nn as nn


RELU = nn.ReLU

class GloReModule(nn.Module):
    def __init__(self, planes, ratio=4, pre_act=False):
        super(GloReModule, self).__init__()
        self.relu = RELU(inplace=True)
        num_gcn_nodes = planes // ratio * 2
        num_gcn_chs = planes // ratio

        if pre_act:
            self.bn_phi = nn.BatchNorm2d(planes)
            self.bn_theta = nn.BatchNorm2d(planes)
        else:
            self.bn_phi, self.bn_theta = None, None

        self.phi = nn.Conv2d(planes, num_gcn_nodes, kernel_size=1, bias=False)
        self.theta = nn.Conv2d(planes, num_gcn_chs, kernel_size=1, bias=False)

        #  Interaction Space
        #  Adjacency Matrix: (-)A_g
        self.bn_adj = nn.BatchNorm1d(num_gcn_chs)
        self.conv_adj = nn.Conv1d(num_gcn_chs, num_gcn_chs, kernel_size=1, bias=False)

        #  State Update Function: W_g
        self.bn_wg = nn.BatchNorm1d(num_gcn_nodes)
        self.conv_wg = nn.Conv1d(num_gcn_nodes, num_gcn_nodes, kernel_size=1, bias=False)

        #  last fc
        self.bn3 = nn.BatchNorm2d(num_gcn_nodes)
        self.conv3 = nn.Conv2d(num_gcn_nodes, planes, kernel_size=1, bias=False)
        if not pre_act:
            self.bn_out = nn.BatchNorm2d(planes)
        else:
            self.bn_out = None

    def to_matrix(self, x):
        n, c, h, w = x.size()
        x = x.view(n, c, -1)
        return x

    def forward(self, x):
        # # # # Projection Space # # # #
        if self.bn_phi is not None and self.bn_theta is not None:
            x_sqz = self.relu(self.bn_phi(x))
            b = self.relu(self.bn_theta(x))
        else:
            x_sqz, b = x, x

        x_sqz = self.phi(x_sqz)
        x_sqz = self.to_matrix(x_sqz)

        b = self.theta(b)
        b = self.to_matrix(b)

        # Project
        z_idt = torch.matmul(x_sqz, b.transpose(1, 2))

        # # # # Interaction Space # # # #
        z = z_idt.transpose(1, 2).contiguous()  # mem copy unavoidable
        z = self.relu(self.bn_adj(z))
        z = self.conv_adj(z)

        z = z.transpose(1, 2).contiguous()
        # Laplacian smoothing: (I - A_g)Z => Z - A_gZ
        z += z_idt
        z = self.relu(self.bn_wg(z))
        z = self.conv_wg(z)

        # # # # Re-projection Space # # # #
        # Re-project
        y = torch.matmul(z, b)

        n, _, h, w = x.size()
        y = y.view(n, -1, h, w)
        y = self.relu(self.bn3(y))
        y = self.conv3(y)

        out = x+y

        if self.bn_out is not None:
            out = self.relu(self.bn_out(out))
        return out


class SEModule(nn.Module):
    def __init__(self, ch, sqz_ratio=16, rm_bias=False):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

        self.fc1 = nn.Conv2d(ch, ch // sqz_ratio, kernel_size=1, bias=(not rm_bias))
        self.relu = RELU(inplace=True)
        self.fc2 = nn.Conv2d(ch // sqz_ratio, ch, kernel_size=1, bias=(not rm_bias))

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class GCBlock(nn.Module):
    def __init__(self,
                 inplanes,
                 ratio,
                 pooling_type='att',
                 fusion_types=('channel_add', )):
        super(GCBlock, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                RELU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                RELU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        out = x
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out


class BNET2d(nn.Module):
    def __init__(self, cfg, width):
        super(BNET2d, self).__init__()
        k = cfg.RPN_STAGE.BACKBONE.BNET_K
        self.bn = nn.BatchNorm2d(width, eps=1e-3, momentum=0.01, affine=False)
        self.bnconv = nn.Conv2d(width, width, k, padding=(k-1)//2, groups=width, bias=True)

    def forward(self, x):
        return self.bnconv(self.bn(x))


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = int(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True)
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out