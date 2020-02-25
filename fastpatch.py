
import torch
from torch.autograd import Function
import fastpatch_cuda as fp_cuda


class FeatPatchFn(Function):
    MaxSize = 200
    NnList = None

    @staticmethod
    def set_maxsize(val):
        FeatPatchFn.MaxSize = val

    @staticmethod
    def set_nnlist(nn_list):
        FeatPatchFn.NnList = nn_list

    @staticmethod
    def forward(ctx, feat):
        patchfeat = fp_cuda.feat_forward(
            feat, FeatPatchFn.NnList, FeatPatchFn.MaxSize)
        return patchfeat

    @staticmethod
    def backward(ctx, grad_patchfeat):
        grad_feat = None

        if ctx.needs_input_grad[0]:
            grad_feat = fp_cuda.feat_backward(
                grad_patchfeat, FeatPatchFn.NnList, FeatPatchFn.MaxSize)
        return grad_feat


feat_patch = FeatPatchFn.apply


def selection_patch(nn_list, maxsize):
    return NotImplementedError
