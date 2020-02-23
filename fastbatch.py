
import torch
from torch.autograd import Function
import fastbatch_cuda as fb_cuda


class FeatBatchFn(Function):
    MaxSize = 200

    @staticmethod
    def set_maxsize(val):
        FeatBatchFn.MaxSize = val

    @staticmethod
    def forward(ctx, feat, nn_list):
        ctx.save_for_backward(nn_list)
        batchfeat = fb_cuda.fb_forward(feat, nn_list)
        return batchfeat

    @staticmethod
    def backward(ctx, grad_batchfeat):
        nn_list, = ctx.saved_tensors
        grad_feat = grad_nn_list = None

        if ctx.needs_input_grad[0]:
            grad_feat = fb_cuda.fb_backward(grad_batchfeat, nn_list)
        return grad_feat, grad_nn_list


feat_batch = FeatBatchFn.apply


def raw_batch(nn_list, maxsize):
    return NotImplementedError
