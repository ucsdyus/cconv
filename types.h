#ifndef TYPES_H_
#define TYPES_H_

#include <vector>
#include <cmath>
#include <torch/extension.h>

namespace fastpatch {

using NeighborList_t = torch::Tensor;  // Ns
using WeightList_t = torch::Tensor;  // Ns x S
using NearNeighbor_t = std::pair<NeighborList, WeightList>;
using NnList_t = std::vector<NearNeighbor_t>;

#define __hd__ __host__ __device__

}  // namespace fastpatch

#endif  // TYPES_H_
