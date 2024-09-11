# Compiler
NVCC := nvcc

# Compiler flags
CFLAGS := -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0

# CUDA flags
CUDAFLAGS := -g -G

# Include directories
WORKSPACE = $(shell pwd)/..

LDFLAGS = -L$(WORKSPACE)/.conda/lib/python3.11/site-packages/torch/lib \
			-ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda -ltorch_python \
          -L/usr/local/cuda/lib64 -lcudart -lcublas -lcusparse -lcurand


INCLUDES = -I$(WORKSPACE)/.conda/lib/python3.11/site-packages/torch/include/THC \
           -I$(WORKSPACE)/.conda/lib/python3.11/site-packages/torch/include \
           -I$(WORKSPACE)/.conda/lib/python3.11/site-packages/torch/include/torch/csrc/api/include \
           -I$(WORKSPACE)/.conda/include/python3.11 \
           -I$(WORKSPACE)


# Source files
SRCS := scan.cu

# Output executable
TARGET := scan

all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(CFLAGS) $(CUDAFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET)