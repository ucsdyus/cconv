import torch
import torch.nn as nn
import torch.nn.init as init
import fastpatch as fp
import math


class Params:
    SpatialSize = None
    SelectMat = None


class CConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(CConv, self).__init__()
        assert Params.SpatialSize is not None, "Set property first"

        self.ch_in = ch_in
        self.ch_out = ch_out
        self.max_size = fp.Params.MaxSize

        self.weight = nn.Parameter(
            torch.Tensor(1, 1, Params.SpatialSize, ch_out * ch_in), requires_grad=True)
        
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, feat_in):
        patch_feat = fp.feat_patch(feat_in).view(-1, self.max_size, self.ch_in, 1)
        patch_weight = torch.matmul(
            Params.SelectMat, self.weight).view(-1, self.max_size, self.ch_out, self.ch_in)
        feat_out = torch.matmul(patch_weight, patch_feat).sum(axis=1).view(-1, self.ch_out, 1)
        return feat_out


def set_property(spatial, nw_list):
    assert fp.Params.MaxSize is not None, "Set fastpatch property first"
    Params.SpatialSize = spatial
    Params.SelectMat = fp.selection_mat_patch(nw_list, spatial)