#include <vector>
#include <torch/extension.h>
#include "types.h"

namespace fastpatch {

torch::Tensor feat_forward(torch::Tensor feat, NnList_t& nn_list);

torch::Tensor feat_backward(torch::Tensor grad_patchfeat, NnList_t& nn_list);

}  // namespace fastpatch


PYBIND11_MODULE(fastpatch_cuda, m) {
    m.doc() = "Fast patch Implementation";

    // Fast patch
    m.def("feat_forward", &fastpatch::feat_forward, "fastpatch forward function");
    m.def("feat_backward", &fastpatch::feat_backward, "fastpatch backward function");
}