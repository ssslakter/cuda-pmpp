#include <utils.cuh>


#define TILE_DIM 32

__global__ void conv1d_k(float *m, float *f, float *out, int f_size, int m_w)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int r = f_size / 2;

    if (idx < m_w)
    {
        float val = 0;
        for (int i = 0; i < 2 * r + 1; i++)
        {
            int f_idx = idx - r + i;
            if (f_idx >= 0 && f_idx < m_w)
            {
                val += m[f_idx] * f[i];
            }
        }
        out[idx] = val;
    }
}

torch::Tensor conv1d(torch::Tensor m, torch::Tensor f)
{
    CHECK_INPUT(m);
    CHECK_INPUT(f);
    int n = m.size(0);
    auto output = torch::zeros(n, m.options());

    conv1d_k<<<cdiv(n, TILE_DIM), TILE_DIM>>>(m.data_ptr<float>(), f.data_ptr<float>(), output.data_ptr<float>(), f.size(0), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}


__global__ void conv1d_shared_k(float *m, float *f, float *out, int f_size, int m_w)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int r = f_size / 2;

    __shared__ float tile[TILE_DIM];
    if (idx < m_w){
        tile[threadIdx.x] = m[idx];
    }
    else{
        tile[threadIdx.x] = 0.;
    }
    __syncthreads();
    if (idx < m_w)
    {
        float val = 0;
        for (int i = 0; i < 2 * r + 1; i++)
        {
            if (threadIdx.x + i >= r &&
                threadIdx.x + i - r < TILE_DIM){
                val += tile[threadIdx.x + i - r] * f[i];
            }
            else{
                int in_row = idx - r + i;
                if (in_row >= 0 && in_row < m_w)
                {
                    val += m[in_row] * f[i];
                }
            }
        }
        out[idx] = val;
    }
}

torch::Tensor conv1d_shared(torch::Tensor m, torch::Tensor f)
{
    CHECK_INPUT(m);
    CHECK_INPUT(f);
    int n = m.size(0);
    auto output = torch::zeros(n, m.options());

    conv1d_k<<<cdiv(n, TILE_DIM), TILE_DIM>>>(m.data_ptr<float>(), f.data_ptr<float>(), output.data_ptr<float>(), f.size(0), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("conv1d", &conv1d, "1D convolution");
    m.def("conv1d_shared", &conv1d_shared, "1D convolution with shared memory");
}
