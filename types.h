#ifndef TYPES_H_
#define TYPES_H_

#include <vector>
#include <cuda.h>
#include <torch/extension.h>

namespace fastpatch {

#define __hd__ __host__ __device__
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")

}  // namespace fastpatch

#endif  // TYPES_H_
