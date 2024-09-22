#include <utils.cuh>

#define SECTION_SIZE 1024

__global__ void Brent_Kung_scan_kernel(float *X, float *Y, unsigned int N)
{
    __shared__ float XY[SECTION_SIZE];
    unsigned int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        XY[threadIdx.x] = X[i];
    if (i + blockDim.x < N)
        XY[threadIdx.x + blockDim.x] = X[i + blockDim.x];
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2)
    {
        __syncthreads();
        unsigned int index = (threadIdx.x + 1) * 2 * stride - 1;
        if (index < SECTION_SIZE)
        {
            XY[index] += XY[index - stride];
        }
    }
    for (int stride = SECTION_SIZE / 4; stride > 0; stride /= 2)
    {
        __syncthreads();
        unsigned int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index + stride < SECTION_SIZE)
        {
            XY[index + stride] += XY[index];
        }
    }
    __syncthreads();
    if (i < N)
        Y[i] = XY[threadIdx.x];
    if (i + blockDim.x < N)
        Y[i + blockDim.x] = XY[threadIdx.x + blockDim.x];
}

template <bool UseP>
__global__ void coarsened_scan_kernel(float *X, float *Y, unsigned int N, float *P = nullptr)
{
    unsigned int S = SECTION_SIZE / blockDim.x;
    __shared__ float XY[SECTION_SIZE];
    unsigned int i = blockIdx.x * SECTION_SIZE + threadIdx.x;
    for (unsigned int j = 0; j < SECTION_SIZE; j += blockDim.x)
    {
        if (i + j < N)
            XY[threadIdx.x + j] = X[i + j];
    }
    __syncthreads();
    // Phase 1. single-threaded scan of each S
    for (unsigned int j = 0; j < S - 1; j++)
    {
        XY[S * threadIdx.x + j + 1] = XY[S * threadIdx.x + j] + XY[S * threadIdx.x + j + 1];
    }
    __syncthreads();
    // Phase 2. Brent-Kung scan of the last element of each S
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2)
    {
        __syncthreads();
        unsigned int index = S * (threadIdx.x + 1) * 2 * stride - 1;
        if (index < SECTION_SIZE)
        {
            XY[index] += XY[index - S * stride];
        }
    }
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        __syncthreads();
        unsigned int index = S * (threadIdx.x + 1) * stride * 2 - 1;
        if (index + stride < SECTION_SIZE)
        {
            XY[index + S * stride] += XY[index];
        }
    }
    __syncthreads();
    // Phase 3. Sum the last element of each S with each element of the next S
    if (threadIdx.x != 0)
    {
        for (unsigned int j = 0; j < S - 1; j++)
        {
            XY[S * threadIdx.x + j] += XY[S * threadIdx.x - 1];
        }
    }
    __syncthreads();
    for (unsigned int j = 0; j < SECTION_SIZE; j += blockDim.x)
    {
        if (i + j < N)
            Y[i + j] = XY[threadIdx.x + j];
    }
    if constexpr (UseP)
    {
        if (threadIdx.x == blockDim.x - 1)
        {
            P[blockIdx.x] = XY[SECTION_SIZE - 1];
        }
    }
}

__global__ void scan_kernel_add(float *P, float *Y, unsigned int N)
{
    unsigned int i = (blockIdx.x + 1) * SECTION_SIZE + threadIdx.x;
    if (i < N)
        Y[i] += P[blockIdx.x];
}

torch::Tensor scan_block(torch::Tensor X, bool is_bk = false)
{
    CHECK_INPUT(X);
    auto Y = torch::zeros_like(X);
    auto N = X.size(0);
    auto num_blocks = cdiv(N, SECTION_SIZE);
    if (is_bk)
        Brent_Kung_scan_kernel<<<num_blocks, SECTION_SIZE / 2>>>(X.data_ptr<float>(), Y.data_ptr<float>(), N);
    else
        coarsened_scan_kernel<false><<<num_blocks, 64>>>(X.data_ptr<float>(), Y.data_ptr<float>(), N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return Y;
}

torch::Tensor scan(torch::Tensor X, bool is_bk = false)
{
    CHECK_INPUT(X);
    auto Y = torch::zeros_like(X);
    auto N = X.size(0);
    auto num_blocks = cdiv(N, SECTION_SIZE);
    if (num_blocks <= 1)
    {
        return scan_block(X, is_bk);
    }
    auto P = torch::zeros({num_blocks}, X.options());
    coarsened_scan_kernel<true><<<num_blocks, 64>>>(X.data_ptr<float>(), Y.data_ptr<float>(), N, P.data_ptr<float>());

    P = (num_blocks > SECTION_SIZE) ? scan(P, is_bk) :scan_block(P, is_bk);
    scan_kernel_add<<<num_blocks - 1, SECTION_SIZE>>>(P.data_ptr<float>(), Y.data_ptr<float>(), N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return Y;
}

// int main()
// {
//     auto x = torch::ones({SECTION_SIZE}).cuda();
//     auto y = scan(x);
//     std::cout << y << std::endl;
//     return 0;
// }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("scan", &scan, "1D scan (prefix sum) of a tensor",
          pybind11::arg("X"), pybind11::arg("is_bk") = false);
}
