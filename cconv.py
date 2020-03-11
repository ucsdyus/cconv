import torch
import torch.nn as nn
import torch.nn.init as init
import cconv.fastpatch as fp
import math


class CConvConfig(object):
    def __init__(self, spatial, max_size):
        self.SpatialSize = spatial
        self.MaxSize = max_size
        self.SelectMat = None

    def update(self, fp_config):
        self.SelectMat = fp.selection_mat_patch(fp_config, self.MaxSize, self.SpatialSize)


class CConv(nn.Module):
    def __init__(self, ch_in, ch_out, cconv_config):
        super(CConv, self).__init__()
        assert cconv_config is not None, "Config cannot be None"

        self.config = cconv_config
        self.ch_in = ch_in
        self.ch_out = ch_out

        self.weight = nn.Parameter(
            torch.Tensor(1, 1, self.config.SpatialSize, ch_out * ch_in), requires_grad=True)
        # print(self.weight.size())

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, fp_config, feat_in):
        # N x M x Cin x 1
        patch_feat = fp.feat_patch(fp_config, feat_in).view(-1, self.config.MaxSize, self.ch_in, 1)
        # N x M x Cout x Cin
        # patch_weight = torch.matmul(
        #     self.config.SelectMat, self.weight).view(-1, self.config.MaxSize, self.ch_out, self.ch_in)
        patch_weight = torch.einsum(
        	'abij,cdjk->abik', (self.config.SelectMat, self.weight)).view(-1, self.config.MaxSize, self.ch_out, self.ch_in)
        # N x Cout x 1
        # feat_out = torch.matmul(patch_weight, patch_feat).sum(axis=1).view(-1, self.ch_out, 1)
        feat_out = torch.einsum('abij,abjk->abik', (patch_weight,patch_feat)).sum(axis=1).view(-1,self.ch_out,1)
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

    def forward(self, fp_config, fixed_in):
        # N x M x Cin x 1
        patch_fixed = fp.fixed_patch(fp_config, fixed_in).view(
            -1, self.config.MaxSize, self.ch_in, 1)
        # N x M x Cout x Cin
        # print(self.config.SelectMat.size())
        # print(self.weight.size())
        # patch_weight = torch.matmul(
        #     self.config.SelectMat, self.weight).view(-1, self.config.MaxSize, self.ch_out, self.ch_in)
        patch_weight = torch.einsum(
        	'abij,cdjk->abik', (self.config.SelectMat, self.weight)).view(-1, self.config.MaxSize, self.ch_out, self.ch_in)
        # N x Cout x 1
        # fixed_out = torch.matmul(patch_weight, patch_fixed).sum(axis=1).view(-1, self.ch_out, 1)
        fixed_out = torch.einsum('abij,abjk->abik', (patch_weight,patch_fixed)).sum(axis=1).view(-1,self.ch_out,1)
        return fixed_out
