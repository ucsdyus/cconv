
import torch
from torch.autograd import Function
import fastpatch_impl as fp_impl


class FpParams:
    MaxSize = None

class FpConfig(object):
    def __init__(self, nn_offset, nn_list, nw_list, grad_nn_offset, grad_nn_list):
        self.nn_offset = nn_offset
        self.nn_list = nn_list
        self.nw_list = nw_list
        self.grad_nn_offset = grad_nn_offset
        self.grad_nn_list = grad_nn_list


class FeatPatchFn(Function):

    @staticmethod
    def forward(ctx, fp_config, feat):
        patchfeat = fp_impl.feat_forward(
            feat, fp_config.nn_offset,
            fp_config.nn_list, FpParams.MaxSize)
        ctx.save_for_backward(fp_config.grad_nn_offset, fp_config.grad_nn_list)
        return patchfeat

    @staticmethod
    def backward(ctx, grad_patchfeat):
        grad_nn_offset, grad_nn_list = ctx.saved_tensors
        grad_feat = None

        # print("start bp fp", ctx.needs_input_grad)
        print(torch.sum(grad_patchfeat != grad_patchfeat))
        if ctx.needs_input_grad[1]:
            grad_feat = fp_impl.feat_backward(
                grad_patchfeat, grad_nn_offset, grad_nn_list, FpParams.MaxSize)
        # print("end bp fp", grad_feat)
        print(torch.sum(grad_feat != grad_feat))
        return None, grad_feat


def set_maxsize(max_size):
    FpParams.MaxSize = max_size


def build_config(nn_offset, nn_list, nw_list, grad_nn_offset, grad_nn_list):
    return FpConfig(nn_offset, nn_list, nw_list, grad_nn_offset, grad_nn_list)


feat_patch = FeatPatchFn.apply


def fixed_patch(fp_config, fixed_in):
    patch_fixed = fp_impl.feat_forward(
        fixed_in, fp_config.nn_offset, fp_config.nn_list, FpParams.MaxSize)
    return patch_fixed


def selection_mat_patch(fp_config, max_size, spatial):
    return fp_impl.get_selection_mat(fp_config.nn_offset, fp_config.nw_list, max_size, spatial)
