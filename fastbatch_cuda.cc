#include <vector>
#include <torch/extension.h>
#include "types.h"

namespace fastbatch {

torch::Tensor fb_forward(torch::Tensor feat, NearNeighbor_t& nn_list);

torch::Tensor fb_backward(torch::Tensor grad_batchfeat, NearNeighbor_t& nn_list);

}  // namespace fastbatch


PYBIND11_MODULE(fastbatch_cuda, m) {
    m.doc() = "Fast Batch Implementation";

    // Fast Batch
    m.def("fb_forward", &fastbatch::fb_forward, "fastbatch forward function");
    m.def("fb_backward", &fastbatch::fb_backward, "fastbatch backward function");
}