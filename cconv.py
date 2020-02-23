import torch
import torch.nn as nn
import fastbatch as fb


class CConv(nn.Module):
    MaxSize = None
    NnList = None
    SelectMat = None

    def __init__(self, spatial, ch_in, ch_out):
        super(CConv, self).__init__()

        self.spatial = spatial
        self.ch_in = ch_in
        self.ch_out = ch_out

        self.weight = nn.Parameter(torch.Tensor(spatial, ch_out * ch_in))
    
    @staticmethod
    def set_geometry(nn_list, maxsize):
        CConv.NnList = nn_list
        CConv.MaxSize = maxsize
        fb.FeatBatchFn.set_maxsize(maxsize)
        CConv.SelectMat = fb.raw_batch(nn_list, maxsize)  # N x Ns x S -> N x M x S

    def forward(self, feat_in):
        batch_feat = fb.feat_batch(feat_in, self.NnList).view(-1, self.MaxSize, self.ch_in, 1)
        batch_weight = torch.matmul(
            self.SelectMat, self.weight).view(-1, self.MaxSize, self.ch_out, self.ch_in)
        feat_out = torch.matmul(batch_weight, batch_feat).sum(axis=1)
        return feat_out
