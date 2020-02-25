#ifndef TYPES_H_
#define TYPES_H_

#include <vector>
#include <cuda.h>
#include <torch/extension.h>

namespace fastpatch {

using Neighbor_t = torch::Tensor;  // Ns
using Weight_t = torch::Tensor;  // Ns x S
using GradNeighbor_t = torch::Tensor;   // Ns x 2

using NnList_t = std::vector<Neighbor_t>;
using NwList_t = std::vector<Weight_t>;
using GradNnList_t = std::vector<GradNeighbor_t>;

#define __hd__ __host__ __device__
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")

}  // namespace fastpatch

#endif  // TYPES_H_
