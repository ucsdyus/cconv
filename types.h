#ifndef TYPES_H_
#define TYPES_H_

#include <vector>
#include <cmath>
#include <torch/extension.h>

namespace fastbatch {

using NeighborList_t = std::vector<torch::Tensor>;
using WeightList_t = std::vector<torch::Tensor>;
using NearNeighbor_t = std::pair<NeighborList, WeightList>;
using NnList_t = std::vector<NearNeighbor_t>;

#define __hd__ __host__ __device__

}  // namespace fastbatch

#endif  // TYPES_H_
