import torch
from pathlib import Path
import os, sys
sys.path.insert(0, '..')
sys.path.insert(0, '.')
from utils import *


print('start compiling')
mod = load_cu_file(Path(__file__).parent/'conv2d.cu')
print('compiled module')
# init convolution and input matrix
k_size = 3
conv = torch.nn.Conv2d(1, 1, k_size, bias=False, padding=k_size//2)
m1 = torch.rand(1000, 2000).contiguous().cuda()
f = conv.weight[0][0].detach().contiguous().cuda()

n_iters = 20
warmup_iters = 5
for i in range(n_iters):

    # start profiling after 10 warmup iterations
    if i == warmup_iters: torch.cuda.cudart().cudaProfilerStart()

    # push range for current iteration
    if i >= warmup_iters: torch.cuda.nvtx.range_push("conv_shared{}".format(i))
    out = mod.conv2d_shared(m1, f)
    if i >= warmup_iters: torch.cuda.nvtx.range_pop()
    
    if i >= warmup_iters: torch.cuda.nvtx.range_push("conv{}".format(i))
    out = mod.conv2d(m1, f)
    if i >= warmup_iters: torch.cuda.nvtx.range_pop()

torch.cuda.cudart().cudaProfilerStop()