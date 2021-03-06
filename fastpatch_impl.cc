#include <vector>
#include <torch/extension.h>
#include "types.h"

namespace fastpatch {

// feat: N x Cin x 1
// offset: N + 1
// nnlist: squeeze(N x Ns)
// return: N x maxsize x Cin x 1
torch::Tensor feat_forward(torch::Tensor feat, torch::Tensor nn_offset, torch::Tensor nn_list, int maxsize);

// grad_patchfeat: N x maxsize x Cin x 1
// grad_nn_offset: N + 1 (should be the same as nn_offset)
// grad_nn_list: squeeze(N x Ns x 2) <grad_v, grad_v_offset: the position of u in nn_list[v]>
// return: N x Cin x 1
torch::Tensor feat_backward(
    torch::Tensor grad_patchfeat, torch::Tensor grad_nn_offset, torch::Tensor grad_nn_list, int maxsize);

// nn_offset: N x maxsize x Cin x 1
// nw_list: squeeze(N x Ns x S)
// return: N x maxsize x 1 x S
torch::Tensor get_selection_mat(torch::Tensor nn_offset, torch::Tensor nw_list, int maxsize, int S);

// // grad_nn_offset (N x 2), grad_nn_list (N x Ns')
// std::pair<torch::Tensor, torch::Tensor> get_grad_nn_list(torch::Tensor nn_offset, torch::Tensor nn_list);
}  // namespace fastpatch


PYBIND11_MODULE(fastpatch_impl, m) {
    m.doc() = "Fast patch Implementation";

    // Fast patch
    m.def("feat_forward", &fastpatch::feat_forward, "fastpatch forward function");
    m.def("feat_backward", &fastpatch::feat_backward, "fastpatch backward function");

    m.def("get_selection_mat", &fastpatch::get_selection_mat, "patch nn_list ot get selection matrix");

    // m.def("get_grad_nn_list", &fastpatch::get_grad_nn_list, "get grad_nn_list for BP");
}