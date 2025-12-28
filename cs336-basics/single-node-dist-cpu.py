import os
import timeit

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def setup(rank, group, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(group, rank=rank, world_size=world_size)

def dist_demo(rank, group, world_size, data_size_mb):
    setup(rank, group, world_size)

    dtype = torch.int32
    bytes_per_element = torch.tensor([], dtype=dtype).element_size()

    num_elements = data_size_mb * 1024 * 1024 // bytes_per_element

    data = torch.randint(
        low=0,
        high=1000,
        size=(num_elements,),
        dtype=dtype
    )

    # data = torch.randint(0, 10, (3,))
    # print(f"rank {rank} data (before all-reduce): {data}")
    dist.all_reduce(data, async_op=False)
    # print(f"rank {rank} data (after all-reduce): {data}")

if __name__  == "__main__":
    # for group in ['gloo', 'nccl']:
    for group in ['gloo']:
        for data_mb in [1, 10, 100]:
            for procs in [2, 4, 6]:
                print(f"bench marking {group} {data_mb}MB procs={procs}")
                execution_time = timeit.timeit(lambda: [mp.spawn(fn=dist_demo, args=(group, procs, data_mb, ), nprocs=procs, join=True)], number=1)

                print(f"Time taken for mp.spawn: {execution_time:.6f} seconds")

                # mp.spawn(group=group, fn=dist_demo, args = (group, procs, data_mb, ), nprocs=procs, join=True)