import torch
import torch.nn as nn
import torch.nn.init as init
import fastpatch as fp
import math


class CConvConfig(object):
    def __init__(self, spatial, max_size):
        self.SpatialSize = spatial
        self.MaxSize = max_size
        self.SelectMat = None

    def update(self, nn_offset, nw_list):
        self.SelectMat = fp.selection_mat_patch(nn_offset, nw_list, self.MaxSize, self.SpatialSize)


class CConv(nn.Module):
    def __init__(self, ch_in, ch_out, cconv_config):
        super(CConv, self).__init__()
        assert cconv_config is not None, "Config cannot be None"

        self.config = cconv_config
        self.ch_in = ch_in
        self.ch_out = ch_out

        self.weight = nn.Parameter(
            torch.Tensor(1, 1, self.config.SpatialSize, ch_out * ch_in), requires_grad=True)

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, feat_in):
        # N x M x Cin x 1
        patch_feat = fp.feat_patch(feat_in).view(-1, self.config.MaxSize, self.ch_in, 1)
        # N x M x Cout x Cin
        patch_weight = torch.matmul(
            self.config.SelectMat, self.weight).view(-1, self.config.MaxSize, self.ch_out, self.ch_in)
        # N x Cout x 1
        feat_out = torch.matmul(patch_weight, patch_feat).sum(axis=1).view(-1, self.ch_out, 1)
        return feat_out


class CConvFixed(nn.Module):
    def __init__(self, ch_in, ch_out, cconv_config):
        super(CConvFixed, self).__init__()
        assert cconv_config is not None, "Config cannot be None"

        self.config = cconv_config
        self.ch_in = ch_in
        self.ch_out = ch_out

        self.weight = nn.Parameter(
            torch.Tensor(1, 1, self.config.SpatialSize, ch_out * ch_in), requires_grad=True)

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, fixed_in):
        # N x M x Cin x 1
        patch_fixed = fp.fixed_patch(fixed_in).view(
            -1, self.config.MaxSize, self.ch_in, 1)
        # N x M x Cout x Cin
        patch_weight = torch.matmul(
            self.config.SelectMat, self.weight).view(-1, self.config.MaxSize, self.ch_out, self.ch_in)
        # N x Cout x 1
        fixed_out = torch.matmul(patch_weight, patch_fixed).sum(axis=1).view(-1, self.ch_out, 1)
        return fixed_out
