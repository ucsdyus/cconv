
#include <vector>
#include <torch/extension.h>
#include "types.h"

namespace fastpatch {

torch::Tensor feat_forward(torch::Tensor feat, NnList_t& nn_list) {
    // Not Implemented
}

torch::Tensor feat_backward(torch::Tensor grad_patchfeat, NnList_t& nn_list) {
    // Not Implemented
}

}  // namespace fastpatch
