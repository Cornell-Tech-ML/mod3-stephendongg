import minitorch
import time
import numpy as np

from minitorch.tensor_ops import TensorBackend

FastTensorBackend = minitorch.TensorBackend(minitorch.FastOps)
GPUBackend = minitorch.TensorBackend(minitorch.CudaOps)


def run_matmul(backend: TensorBackend, size: int = 16) -> None:
    """Perform batch matrix multiplication using the specified backend.

    This function generates two random tensors of shape `(batch_size, size, size)`
    and performs matrix multiplication using the given backend. The result is
    computed but not returned.

    Args:
    ----
        backend: The backend to use for tensor operations. This can be a CPU-based
                 or GPU-based backend provided by `minitorch`.
        size (int, optional): The size of the square matrices. Defaults to 16.

    Returns:
    -------
        None: The function performs matrix multiplication but does not return a value.

    Notes:
    -----
    - The `batch_size` is fixed at 2.
    - This function is primarily intended to test the performance of the backend.

    """
    batch_size = 2
    x = minitorch.rand((batch_size, size, size), backend=backend)
    y = minitorch.rand((batch_size, size, size), backend=backend)
    _ = x @ y


if __name__ == "__main__":
    # Warmup
    run_matmul(FastTensorBackend)
    run_matmul(GPUBackend)

    ntrials = 3
    times = {}

    for size in [64, 128, 256, 512, 1024]:
        print(f"Running size {size}")
        times[size] = {}
        fast_times = []
        gpu_times = []

        for _ in range(ntrials):
            # Fast backend timing
            start_fast = time.time()
            run_matmul(FastTensorBackend, size)
            end_fast = time.time()

            # GPU backend timing
            start_gpu = time.time()
            run_matmul(GPUBackend, size)
            end_gpu = time.time()

            fast_time = end_fast - start_fast
            gpu_time = end_gpu - start_gpu

            fast_times.append(fast_time)
            gpu_times.append(gpu_time)

        times[size]["fast"] = np.mean(fast_times)
        times[size]["gpu"] = np.mean(gpu_times)

        print(times[size])
        print()

    print("Timing summary")
    for size, stimes in times.items():
        print(f"Size: {size}")
        for b, t in stimes.items():
            print(f"    {b}: {t:.5f}")
