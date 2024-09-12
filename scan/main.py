import torch
from pathlib import Path
import os, sys
sys.path.insert(0, '..')
sys.path.insert(0, '.')
from utils import *


print('start compiling')
mod = load_cu_file(Path(__file__).parent/'scan.cu')
print('compiled module')
# init convolution and input matrix
x = torch.ones(2100).contiguous().cuda()
# res = mod.scan(x)
# print(res)

n_iters = 20
warmup_iters = 5
for i in range(n_iters):

    # start profiling after 10 warmup iterations
    if i == warmup_iters: torch.cuda.cudart().cudaProfilerStart()

    # push range for current iteration
    if i >= warmup_iters: torch.cuda.nvtx.range_push("scan_coal{}".format(i))
    out = mod.scan(x, False)
    if i >= warmup_iters: torch.cuda.nvtx.range_pop()
    
    if i >= warmup_iters: torch.cuda.nvtx.range_push("scan_bk{}".format(i))
    out = mod.scan(x, True)
    if i >= warmup_iters: torch.cuda.nvtx.range_pop()

torch.cuda.cudart().cudaProfilerStop()