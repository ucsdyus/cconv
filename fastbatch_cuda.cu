
#include <torch/extension.h>
#include "types.h"

namespace fastbatch {

torch::Tensor fb_forward(torch::Tensor feat, NearNeighbor_t& nn_list) {
    // Not Implemented
}

torch::Tensor fb_backward(torch::Tensor grad_batchfeat, NearNeighbor_t& nn_list) {
    // Not Implemented
}

}  // namespace fastbatch
