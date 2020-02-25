
#include <cuda.h>
#include <torch/extension.h>
#include "types.h"

#ifndef THREAD_NUM
#define THREAD_NUM 32
#endif  // THREAD_NUM

namespace fastpatch {
namespace {

#define CHECK_RUNTIME_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

}  // namespace

__global__ void feat_forward_kernel(int N, int maxsize, int Cin,
    const NnList_t*  __restrict__ nn_list, const float* __restrict__  feat_data,
    float* __restrict__  featpatch_data) {

        int u = blockIdx.x;  // bi
        int N_RW = blockDim.x;
        int PATCH_STRIDE = maxsize * Cin;

        int Ns = torch::size(nn_list[u], 0);
        int* nn = nn_list[u].data_ptr<int>();

        int ti = threadIdx.x;
        
        for (int i = ti; i < Ns; i += N_RW) {
            int v = nn[i];
            cudaMemcpy(featpatch_data + u * PATCH_STRIDE + i * Cin, feat_data + v * Cin,
                Cin * sizeof(float), cudaMemcpyDeviceToDevice);
        }
}

__global__ void feat_backward_kernel(int N, int maxsize, int Cin,
    const GradNnList_t*  __restrict__ grad_nn_list, const float* __restrict__  grad_patchfeat,
    float* __restrict__  grad_feat) {

        int u = blockIdx.x;  // bi
        int N_RW = blockDim.x;
        int PATCH_STRIDE = maxsize * Cin;

        int Ns = torch::size(grad_nn_list[u], 0);
        int* grad_nn =  grad_nn_list[u]->data_ptr<int>();  // Ns x 2

        int ti = threadIdx.x;
        int tj = threadIdx.y;
        
        for (int i = ti; i < Ns; i += N_RW) {
            int v = grad_nn[i * 2];
            int offset = grad_nn[i * 2 + 1];
            grad_feat[u * Cin + tj] += grad_patchfeat[v * PATCH_STRIDE + offset * Cin + tj];
        }
}


torch::Tensor feat_forward(torch::Tensor feat, NnList_t& nn_list, int maxsize) {
    // Not Implemented
    CHECK_CUDA(feat);
    
    int N = nn_list.size();
    int Cin = torch::size(feat, 1);

    torch::Tensor patchfeat = torch::zeros({N, maxsize, Cin, 1}, feat.options());

    const dim3 block(THREAD_NUM);
    const dim3 grid(N);
    feat_forward_kernel<<<grid, block>>>(
        N, maxsize, Cin, nn_list.data(),
        feat.data_ptr<float>(), patchfeat.data_ptr<float>());
    
    CHECK_RUNTIME_ERROR(cudaPeekAtLastError());
    return patchfeat;
}


torch::Tensor feat_backward(torch::Tensor grad_patchfeat, GradNnList_t& grad_nn_list, int maxsize) {
    CHECK_CUDA(grad_patchfeat);

    int N = grad_nn_list.size();
    int Cin = torch::size(feat, 2);

    torch::Tensor grad_feat = torch::zeros({N, Cin, 1}, feat.options());

    const dim3 block(THREAD_NUM, Cin);
    const dim3 grid(N);
    feat_backward_kernel<<<grid, block>>>(
        N, maxsize, Cin, grad_nn_list.data(),
        grad_patchfeat.data_ptr<float>(), grad_feat.data_ptr<float>());
    
    CHECK_RUNTIME_ERROR(cudaPeekAtLastError());
    return grad_feat;
}


__global__ void get_selection_mat_kernel(
    int maxsize, int S, const NwList_t* __restrict__ nw_list, float* __restrict__ select_mat) {
        int bi = blockIdx.x;
        int STRIDE = maxsize * S;

        int Ns = torch::size(nw_list[bi], 0);
        float* nw = nw_list[bi].data_ptr<float>();

        cudaMemcpy(select_mat + bi * STRIDE, nw,
            Ns * S * sizeof(float), cudaMemcpyDeviceToDevice);
}


torch::Tensor get_selection_mat(NnList_t& nn_list, NwList_t& nw_list, int maxsize) {
    CHECK_CUDA(nn_list[0])/;
    int N = nn_list.size();
    int S =  torch::size(nw_list[0], 1);

    torch::Tensor select_mat = torch::zeros({N, maxsize, S}, feat.options());

    const dim3 block(1);
    const dim3 grid(N);
    get_selection_mat_kernel<<<grid, block>>>(
        N, maxsize, Cin, grad_nn_list.data(),
        grad_patchfeat.data_ptr<float>(), grad_feat.data_ptr<float>());

    
    CHECK_RUNTIME_ERROR(cudaPeekAtLastError());
    return select_mat;
}


GradNnList_t grad_nn_list(NnList_t& nn_list) {
    int N = nn_list.size();
    std::vector<std::vector<int>> grad_nn_v(N);
    std::vector<std::vector<int>> grad_nn_offset(N);

    for (int u = 0; u < N; ++u) {
        int Ns = torch::size(nn_list[u], 0);
        int* nn = nn_list[u].data_ptr<int>();
        for (int j = 0; j < Ns; ++j) {
            grad_nn_v[nn[j]].push_back(u);
            grad_nn_offset[nn[j]].push_back(j);
        }
    }
    GradNnList_t grad_nn_list;
    grad_nn_list.reserve(N);
    for (int u = 0; u < N; ++u) {
        int Ns = grad_nn_v[u].size();
        torch::Tensor grad_nn = torch::zeros({Ns, 2}, nn_list[0].options());
        grad_nn_list.push_back(grad_nn);
        
        int* grad_nn_ptr = grad_nn.data_ptr<int>();
        for (int i = 0; i < grad_nn_v.size(); ++i) {
            grad_nn_ptr[i * 2] = grad_nn_v[i];
            grad_nn_ptr[i * 2 + 1] = grad_nn_offset[i];
        }
    }
    return grad_nn_list;
}

}  // namespace fastpatch
