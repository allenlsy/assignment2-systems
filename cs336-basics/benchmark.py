from cs336_basics import model
import torch
from jaxtyping import Int
from torch import Tensor
import timeit

vocab_size = 10000
context_length = 10000
d_model = 1000
num_layers = 10
num_heads = 10
d_ff = 1000
batch_size = 1
sequence_length = context_length

x: Int[Tensor, "... sequence_length"] = torch.randint(
    low=0,
    high=vocab_size,
    size=(batch_size, sequence_length),
    dtype=torch.int64,
)

m = model.BasicsTransformerLM(
    vocab_size=vocab_size,
    context_length=context_length,
    d_model=d_model,
    num_layers=num_layers,
    num_heads=num_heads,
    d_ff=d_ff,
    rope_theta=10000.0,
)


# Time the execution using timeit
execution_time = timeit.timeit(lambda: [m.forward(x), torch.cuda.synchronize()], number=1)
print(f"Time taken for m.forward(): {execution_time:.6f} seconds")
