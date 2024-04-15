from functools import partial
import torch

def trace_handler(prof, save_path):
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=-1))
    prof.export_chrome_trace(save_path + str(prof.step_num) + ".json")

def profile(f, inp, times=10, save_path="/tmp/test_trace_"):
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],

        # In this example with wait=1, warmup=1, active=2, repeat=1,
        # profiler will skip the first step/iteration,
        # start warming up on the second, record
        # the third and the forth iterations,
        # after which the trace will become available
        # and on_trace_ready (when set) is called;
        # the cycle repeats starting with the next step

        schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=2,
            repeat=1),
        on_trace_ready=partial(trace_handler,save_path= save_path)
        # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
        # used when outputting for tensorboard
        ) as p:
            for _ in range(times): 
                f(inp)
                # send a signal to the profiler that the next iteration has started
                p.step()