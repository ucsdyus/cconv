#include <vector>
#include <torch/extension.h>
#include "types.h"

namespace fastpatch {

torch::Tensor feat_forward(torch::Tensor feat, NnList_t& nn_list, int maxsize);

torch::Tensor feat_backward(torch::Tensor grad_patchfeat, GradNnList_t& grad_nn_list, int maxsize);

torch::Tensor get_selection_mat(NnList_t& nn_list, int maxsize);

GradNnList_t get_grad_nn_list(NnList_t& nn_list);
}  // namespace fastpatch


PYBIND11_MODULE(fastpatch_cuda, m) {
    m.doc() = "Fast patch Implementation";

    // Fast patch
    m.def("feat_forward", &fastpatch::feat_forward, "fastpatch forward function");
    m.def("feat_backward", &fastpatch::feat_backward, "fastpatch backward function");

    m.def("get_selection_mat", &fastpatch::get_selection_mat, "patch nn_list ot get selection matrix");

    m.def("get_grad_nn_list", &fastpatch::get_grad_nn_list, "get grad_nn_list for BP");
}