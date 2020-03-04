
import torch
from torch.autograd import Function
import fastpatch_impl as fp_impl


class FeatPatchParams:
    MaxSize = None
    NnOffset = None
    NnList = None

    GradNnOffset = None
    GradNnList = None


class FixedPatchParams:
    MaxSize = None
    NnOffset = None
    NnList = None


class FeatPatchFn(Function):

    @staticmethod
    def forward(ctx, feat):
        patchfeat = fp_impl.feat_forward(
            feat, FeatPatchParams.NnOffset,
            FeatPatchParams.NnList, FeatPatchParams.MaxSize)
        return patchfeat

    @staticmethod
    def backward(ctx, grad_patchfeat):
        grad_feat = None

        if ctx.needs_input_grad[0]:
            grad_feat = fp_impl.feat_backward(
                grad_patchfeat, FeatPatchParams.GradNnOffset,
                FeatPatchParams.GradNnList, FeatPatchParams.MaxSize)
        return grad_feat


def update_feat_config(max_size, nn_offset, nn_list, grad_nn_offset, grad_nn_list):
    FeatPatchParams.MaxSize = max_size
    FeatPatchParams.NnOffset = nn_offset
    FeatPatchParams.NnList = nn_list

    FeatPatchParams.GradNnOffset = grad_nn_offset
    FeatPatchParams.GradNnList = grad_nn_list


def update_fixed_config(max_size, nn_offset, nn_list):
    FixedPatchParams.MaxSize = max_size
    FixedPatchParams.NnOffset = nn_offset
    FixedPatchParams.NnList = nn_list


feat_patch = FeatPatchFn.apply


def fixed_patch(fixed_in):
    patch_fixed = fp_impl.feat_forward(
        fixed_in, FixedPatchParams.NnOffset,
        FixedPatchParams.NnList, FixedPatchParams.MaxSize)
    return patch_fixed


def selection_mat_patch(nn_offset, nw_list, max_size, spatial):
    return fp_impl.get_selection_mat(nn_offset, nw_list, max_size, spatial)
