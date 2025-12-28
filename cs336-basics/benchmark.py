from cs336_basics import model
import torch
from jaxtyping import Int
from torch import Tensor
import timeit
import sys


def benchmark_transformer(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, batch_size=1):
    """
    Benchmark the performance of a BasicsTransformerLM model with given specifications.
    
    Args:
        vocab_size: Size of the vocabulary
        context_length: Maximum context length for the model
        d_model: Dimension of the model embeddings
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        d_ff: Dimension of the feed-forward layer
        batch_size: Batch size for benchmarking (default: 1)
        
    Returns:
        Execution time in seconds
    """
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
    
    return execution_time


specs = {
    "small": {
        "d_model": 768,
        "d_ff": 3072,
        "num_layers": 12,
        "num_heads": 12,
    },
    "medium": {
        "d_model": 1024,
        "d_ff": 4096,
        "num_layers": 24,
        "num_heads": 16,
    },
    "large": {
        "d_model": 1280,
        "d_ff": 5120,
        "num_layers": 36,
        "num_heads": 20,
    },
    "xl": {
        "d_model": 1600,
        "d_ff": 6400,
        "num_layers": 48,
        "num_heads": 25,
    },
    "2.7B": {
        "d_model": 2560,
        "d_ff": 10240,
        "num_layers": 32,
        "num_heads": 32,
    },
}

# Benchmark all model specifications
if __name__ == "__main__":
    # Fixed parameters for all benchmarks
    vocab_size = 10000
    context_length = 10000
    batch_size = 1
    
    print("Benchmarking different model specifications:")
    print("=" * 50)
    print(f"Found {len(specs)} model specifications to benchmark")
    
    # Loop through each specification and benchmark
    for model_name, model_spec in specs.items():
        print(f"\nBenchmarking {model_name} model:")
        print(f"  d_model: {model_spec['d_model']}")
        print(f"  d_ff: {model_spec['d_ff']}")
        print(f"  num_layers: {model_spec['num_layers']}")
        print(f"  num_heads: {model_spec['num_heads']}")
        
        try:
            benchmark_transformer(
                vocab_size=vocab_size,
                context_length=context_length,
                d_model=model_spec["d_model"],
                num_layers=model_spec["num_layers"],
                num_heads=model_spec["num_heads"],
                d_ff=model_spec["d_ff"],
                batch_size=batch_size
            )
        except BaseException as e:
            print(f"  Error benchmarking {model_name}: {str(e)}")
            # Also flush the output in case of an error
            sys.stdout.flush()
        
        print("-" * 50)
    
    print("\nBenchmarking completed!")
    sys.stdout.flush()

