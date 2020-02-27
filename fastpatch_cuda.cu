
#include <cuda.h>
#include <cuda_runtime.h>
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

__global__ void feat_forward_kernel(int maxsize, int Cin, 
    const int* __restrict__ nn_offset, const int*  __restrict__ nn_list, const float* __restrict__  feat_data,
    float* __restrict__  patchfeat_data) {

        int u = blockIdx.x;  // bi
        int N_RW = blockDim.x;
        int PATCH_STRIDE = maxsize * Cin;

        int Ns = nn_offset[u + 1] - nn_offset[u];
        const int* nn = nn_list + nn_offset[u];  // Ns
        

        int ti = threadIdx.x;
        int tj = threadIdx.y;
        for (int i = ti; i < Ns; i += N_RW) {
            int v = nn[i];
            patchfeat_data[u * PATCH_STRIDE + i * Cin + tj] = feat_data[v * Cin + tj];
        }
}

__global__ void feat_backward_kernel(int maxsize, int Cin,
    const int* __restrict__ grad_nn_offset, const int*  __restrict__ grad_nn_list,
    const float* __restrict__  grad_patchfeat, float* __restrict__  grad_feat) {

        int u = blockIdx.x;  // bi
        int N_RW = blockDim.x;
        int PATCH_STRIDE = maxsize * Cin;

        int Ns = grad_nn_offset[u + 1] - grad_nn_offset[u];
        const int* grad_nn = grad_nn_list + grad_nn_offset[u] * 2;  // Ns x 2
        
        int ti = threadIdx.x;
        int tj = threadIdx.y;
        for (int i = ti; i < Ns; i += N_RW) {
            int v = grad_nn[i * 2];
            int v_offset = grad_nn[i * 2 + 1];
            // printf("u, Ns, i, v, v_offset = %d %d %d %d %d\n", u, Ns, i, v, v_offset);
            // use atomicAdd instead of +=
            atomicAdd(grad_feat + u * Cin + tj, grad_patchfeat[v * PATCH_STRIDE + v_offset * Cin + tj]);
        }
}

// feat: N x Cin x 1
// offset: N + 1
// nnlist: squeeze(N x Ns)
torch::Tensor feat_forward(torch::Tensor feat, torch::Tensor nn_offset, torch::Tensor nn_list, int maxsize) {
    // Not Implemented
    CHECK_CUDA(feat);
    CHECK_CUDA(nn_offset);
    CHECK_CUDA(nn_list);
    
    int N = torch::size(feat, 0);
    int Cin = torch::size(feat, 1);

    torch::Tensor patchfeat = torch::zeros({N, maxsize, Cin, 1}, feat.options());

    const dim3 block(THREAD_NUM, Cin);
    const dim3 grid(N);
    feat_forward_kernel<<<grid, block>>>(
        maxsize, Cin, nn_offset.data_ptr<int>(), nn_list.data_ptr<int>(),
        feat.data_ptr<float>(), patchfeat.data_ptr<float>());
    
    CHECK_RUNTIME_ERROR(cudaPeekAtLastError());
    return patchfeat;
}


torch::Tensor feat_backward(
    torch::Tensor grad_patchfeat, torch::Tensor grad_nn_offset, torch::Tensor grad_nn_list, int maxsize) {
    CHECK_CUDA(grad_patchfeat);
    CHECK_CUDA(grad_nn_offset);
    CHECK_CUDA(grad_nn_list);

    int N = torch::size(grad_patchfeat, 0);
    int Cin = torch::size(grad_patchfeat, 2);  // N x maxsize x Cin x 1

    torch::Tensor grad_feat = torch::zeros({N, Cin, 1}, grad_patchfeat.options());

    const dim3 block(THREAD_NUM, Cin);
    const dim3 grid(N);
    feat_backward_kernel<<<grid, block>>>(
        maxsize, Cin,
        grad_nn_offset.data_ptr<int>(), grad_nn_list.data_ptr<int>(),
        grad_patchfeat.data_ptr<float>(), grad_feat.data_ptr<float>());
    
    CHECK_RUNTIME_ERROR(cudaPeekAtLastError());
    return grad_feat;
}


__global__ void get_selection_mat_kernel(int maxsize, int S, 
    const int* __restrict__ nn_offset, const float* __restrict__ nw_list, float* __restrict__ select_mat) {
        int u = blockIdx.x;  // bi
        int N_RW = blockDim.x;
        int STRIDE = maxsize * S;

        int Ns = nn_offset[u + 1] - nn_offset[u];
        const float* nw = nw_list + nn_offset[u] * S;  // nw_list N x Ns x S

        int ti = threadIdx.x;
        int tj = threadIdx.y;
        for (int v = ti; v < Ns; v += N_RW) {
            select_mat[u * STRIDE + v * S + tj] = nw[v * S + tj];
        }
}


torch::Tensor get_selection_mat(int S,  torch::Tensor nn_offset, torch::Tensor nw_list, int maxsize) {
    CHECK_CUDA(nn_offset);
    CHECK_CUDA(nw_list);

    int N = torch::size(nn_offset, 0) - 1;
    torch::Tensor select_mat = torch::zeros({N, maxsize, 1, S}, nw_list.options());

    const dim3 block(THREAD_NUM, S);
    const dim3 grid(N);
    get_selection_mat_kernel<<<grid, block>>>(
        maxsize, S, nn_offset.data_ptr<int>(), nw_list.data_ptr<float>(),
        select_mat.data_ptr<float>());
    
    CHECK_RUNTIME_ERROR(cudaPeekAtLastError());
    return select_mat;
}


std::pair<torch::Tensor, torch::Tensor> get_grad_nn_list(torch::Tensor nn_offset, torch::Tensor nn_list) {
    int N = torch::size(nn_offset, 0) - 1;

    std::vector<std::vector<int>> grad_nn_v(N);
    std::vector<std::vector<int>> grad_v_offset(N);

    const int* nn_offset_ptr = nn_offset.data_ptr<int>();
    const int* nn_list_ptr = nn_list.data_ptr<int>();

    for (int u = 0; u < N; ++u) {
        int Ns = nn_offset_ptr[u + 1] - nn_offset_ptr[u];
        const int* nn = nn_list_ptr + nn_offset_ptr[u];

        for (int j = 0; j < Ns; ++j) {
            grad_nn_v[nn[j]].push_back(u);
            grad_v_offset[nn[j]].push_back(j);
        }
    }
    torch::Tensor grad_nn_offset = torch::zeros_like(nn_offset);
    int* grad_nn_offset_ptr = grad_nn_offset.data_ptr<int>();
    for (int i = 1; i <= N; ++i) {
        grad_nn_offset_ptr[i] = grad_nn_offset_ptr[i - 1] + grad_nn_v[i - 1].size();
    }

    torch::Tensor grad_nn_list = torch::zeros(grad_nn_offset_ptr[N] * 2, nn_list.options());  // N x Ns x 2
    int* grad_nn_list_ptr = grad_nn_list.data_ptr<int>();
    for (int u = 0; u < N; ++u) {
        int start = grad_nn_offset_ptr[u];
        int Ns = grad_nn_offset_ptr[u + 1] - grad_nn_offset_ptr[u];
        for (int i = 0; i < Ns; ++i) {
            grad_nn_list_ptr[(start + i) * 2] = grad_nn_v[u][i];
            grad_nn_list_ptr[(start + i) * 2 + 1] = grad_v_offset[u][i];
        }
    }
    return {grad_nn_offset, grad_nn_list};
}

}  // namespace fastpatch
