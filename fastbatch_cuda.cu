
#include <vector>
#include <torch/extension.h>
#include "types.h"

namespace fastbatch {

torch::Tensor fb_forward(torch::Tensor feat, NnList_t& nn_list) {
    // Not Implemented
}

torch::Tensor fb_backward(torch::Tensor grad_batchfeat, NnList_t& nn_list) {
    // Not Implemented
}

}  // namespace fastbatch
