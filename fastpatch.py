
import torch
from torch.autograd import Function
import fastpatch_cuda as fp_cuda


class FeatPatchFn(Function):
    MaxSize = 200

    @staticmethod
    def set_maxsize(val):
        FeatPatchFn.MaxSize = val

    @staticmethod
    def forward(ctx, feat, nn_list):
        ctx.save_for_backward(nn_list)
        patchfeat = fp_cuda.feat_forward(feat, nn_list)
        return patchfeat

    @staticmethod
    def backward(ctx, grad_patchfeat):
        nn_list, = ctx.saved_tensors
        grad_feat = grad_nn_list = None

        if ctx.needs_input_grad[0]:
            grad_feat = fp_cuda.feat_backward(grad_patchfeat, nn_list)
        return grad_feat, grad_nn_list


feat_patch = FeatPatchFn.apply


def selection_patch(nn_list, maxsize):
    return NotImplementedError
