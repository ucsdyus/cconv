
import torch
from torch.autograd import Function
import fastpatch_impl as fp_impl


class Params:
    MaxSize = None

    NnOffset = None
    NnList = None

    GradNnOffset = None
    GradNnList = None


class FeatPatchFn(Function):

    @staticmethod
    def forward(ctx, feat):
        patchfeat = fp_impl.feat_forward(
            feat, Params.NnOffset, Params.NnList, Params.MaxSize)
        return patchfeat

    @staticmethod
    def backward(ctx, grad_patchfeat):
        grad_feat = None

        if ctx.needs_input_grad[0]:
            grad_feat = fp_impl.feat_backward(
                grad_patchfeat, Params.GradNnOffset, Params.GradNnList,
                Params.MaxSize)
        return grad_feat


def set_property(max_size, nn_offset, nn_list, grad_nn_offset, grad_nn_list):
    Params.MaxSize = max_size

    Params.NnOffset = nn_offset
    Params.NnList = nn_list

    Params.GradNnOffset = grad_nn_offset
    Params.GradNnList = grad_nn_list


feat_patch = FeatPatchFn.apply


def selection_mat_patch(nw_list, spatial):
    return fp_impl.get_selection_mat(Params.NnOffset, nw_list, Params.MaxSize, spatial)
