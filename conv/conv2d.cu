#include <utils.cuh>

#define TILE_DIM 32
#define FILTER_RADIUS 1
__constant__ float F_c[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];
__global__ void conv2d_shared_k(float* m, float* out, int m_h, int m_w) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int r = FILTER_RADIUS;
    
    __shared__ float tile[TILE_DIM][TILE_DIM];
    if (row < m_h && col < m_w) {
        tile[threadIdx.y][threadIdx.x] = m[row * m_w + col];
    }
    else{
        tile[threadIdx.y][threadIdx.x] = 0.;
    }
    __syncthreads();
    if (row < m_h && col < m_w) {
        float val = 0;
        for (int i = 0; i < 2*r+1; i++) {
            for (int j = 0; j < 2*r+1; j++) {
                if(threadIdx.x+j>=r && 
                   threadIdx.x-r+j<TILE_DIM && 
                   threadIdx.y+i>=r && 
                   threadIdx.y-r+i<TILE_DIM){
                    val += tile[threadIdx.y+i-r][threadIdx.x+j-r] * F_c[i][j];
                }
                else{
                    int in_row = row - r + i;
                    int in_col = col - r + j;
                    if (in_row >= 0 && in_row < m_h && in_col >= 0 && in_col < m_w) {
                    val += m[in_row * m_w + in_col] * F_c[i][j];
                }
                }
            }
        }
        out[row * m_w + col] = val;
    }
}

torch::Tensor conv2d_shared(torch::Tensor m, torch::Tensor f) {
    CHECK_INPUT(m); CHECK_INPUT(f);
    TORCH_CHECK(f.size(0)==2*FILTER_RADIUS+1 && f.size(1)==2*FILTER_RADIUS+1, 
    "Filter size must be 2*FILTER_RADIUS+1 x 2*FILTER_RADIUS+1");
    int h = m.size(0);
    int w = m.size(1);
    auto output = torch::zeros({h, w}, m.options());
    cudaMemcpyToSymbol(F_c, f.data_ptr<float>(), (2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1)*sizeof(float));

    dim3 tpb(TILE_DIM,TILE_DIM);
    dim3 blocks(cdiv(w, tpb.x), cdiv(h, tpb.y));
    conv2d_shared_k<<<blocks, tpb>>>(
        m.data_ptr<float>(), output.data_ptr<float>(), h, w);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

__global__ void conv2d_k(float* m, float* f, float* out, int f_size, int m_h, int m_w) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int r = f_size/2;

    if (row < m_h && col < m_w) {
        float val = 0;
        for (int i = 0; i < 2*r+1; i++) {
            for (int j = 0; j < 2*r+1; j++) {
                int in_row = row - r + i;
                int in_col = col - r + j;
                if (in_row >= 0 && in_row < m_h && in_col >= 0 && in_col < m_w) {
                    val += m[in_row * m_w + in_col] * f[i*f_size+j];
                }
            }
        }
        out[row * m_w + col] = val;
    }
}

torch::Tensor conv2d(torch::Tensor m, torch::Tensor f) {
    CHECK_INPUT(m); CHECK_INPUT(f);
    int h = m.size(0);
    int w = m.size(1);
    auto output = torch::zeros({h, w}, m.options());

    dim3 tpb(16,16);
    dim3 blocks(cdiv(w, tpb.x), cdiv(h, tpb.y));
    conv2d_k<<<blocks, tpb>>>(
        m.data_ptr<float>(), f.data_ptr<float>(), output.data_ptr<float>(), f.size(0), h, w);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("conv2d_shared", &conv2d_shared, "2D convolution with shared memory");
  m.def("conv2d", &conv2d, "2D convolution");
}
