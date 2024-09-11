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


__global__ void coarsened_scan_kernel(float *X, float *Y, unsigned int N)
{
    unsigned int S = SECTION_SIZE/blockDim.x;
    __shared__ float XY[SECTION_SIZE];
    unsigned int i = blockIdx.x * SECTION_SIZE + threadIdx.x;
    for (unsigned int j = 0; j < S; j++)
    {
        if (i+j*blockDim.x < N)
            XY[threadIdx.x+j*blockDim.x] = X[i+j*blockDim.x];
    }
    __syncthreads();
    // single-threaded scan of each S
    for (unsigned int j = 0; j < S-1; j++)
    {
        XY[S*threadIdx.x+j+1] = XY[S*threadIdx.x+j] + XY[S*threadIdx.x+j+1]; 
    }
    __syncthreads();
    for (unsigned int j = 0; j < S; j++)
    {
        if (i+j*blockDim.x < N)
            Y[i+j*blockDim.x] = XY[threadIdx.x+j*blockDim.x];
    }
}

torch::Tensor scan(torch::Tensor X)
{
    // CHECK_INPUT(X);
    auto Y = torch::zeros_like(X);
    auto N = X.size(0);
    auto num_blocks = cdiv(N, SECTION_SIZE);
    coarsened_scan_kernel<<<num_blocks, 64>>>(X.data_ptr<float>(), Y.data_ptr<float>(), N);
    // C10_CUDA_KERNEL_LAUNCH_CHECK();
    return Y;
}

int main()
{
    auto x = torch::ones({SECTION_SIZE}).cuda();
    auto y = scan(x);
    std::cout << y << std::endl;
    return 0;
}

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
// {
//     m.def("scan", &scan, "1D scan (prefix sum) of a tensor");
// }
