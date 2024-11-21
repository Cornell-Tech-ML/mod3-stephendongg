# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs: Any) -> Fn:
    """JIT-compile a function for use on CUDA devices.

    Args:
    ----
        fn (Fn): The function to compile for CUDA.
        **kwargs: Additional arguments passed to the Numba `jit` function.

    Returns:
    -------
        Fn: The compiled CUDA device function.

    """
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn: Fn, **kwargs: Any) -> FakeCUDAKernel:
    """JIT-compile a CUDA kernel function.

    Args:
    ----
        fn: The function to compile as a CUDA kernel.
        **kwargs: Additional arguments passed to the Numba `jit` function.

    Returns:
    -------
        FakeCUDAKernel: The compiled CUDA kernel.

    """
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """Create a CUDA map operation for element-wise transformations.

        This method compiles a single-argument function `fn` for execution on CUDA.
        The compiled function is applied element-wise to a tensor using a CUDA kernel.

        Args:
        ----
            fn (Callable[[float], float]): A function that maps a float to a float.

        Returns:
        -------
            MapProto: A callable that applies the transformation to a tensor.

        """
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """Create a CUDA zip operation for element-wise transformations on two tensors.

        This method compiles a binary function `fn` for execution on CUDA. The compiled
        function is applied element-wise to two tensors using a CUDA kernel.

        Args:
        ----
            fn (Callable[[float, float], float]): A binary function that maps two floats
                                                to a float.

        Returns:
        -------
            Callable[[Tensor, Tensor], Tensor]: A callable that applies the transformation
                                                to two tensors.

        """
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """Create a CUDA reduce operation for tensor reduction along a dimension.

        This method compiles a binary reduction function `fn` for execution on CUDA.
        The compiled function reduces a tensor along a specified dimension using the
        CUDA kernel.

        Args:
        ----
            fn (Callable[[float, float], float]): A binary function for reduction.
            start (float): The initial value for the reduction operation.

        Returns:
        -------
            Callable[[Tensor, int], Tensor]: A callable that performs the reduction on
                                            the tensor.

        """
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 512
            blockspergrid = out_a.size

            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Perform matrix multiplication on two tensors using CUDA.

        This method supports 2D matrix multiplication or batched 3D matrix multiplication.
        Shared memory and CUDA blocks are used for efficient computation.

        Args:
        ----
            a (Tensor): The first input tensor.
            b (Tensor): The second input tensor.

        Returns:
        -------
            Tensor: The resulting tensor from the matrix multiplication.

        Notes:
        -----
        - Both input tensors are reshaped to 3D for compatibility with batched operations.
        - The function ensures the matrix dimensions match the required shape for multiplication.

        """
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        # Ensure thread index is within bounds
        if i >= out_size:
            return

        # Compute multi-dimensional index for output tensor
        to_index(i, out_shape, out_index)

        broadcast_index(out_index, out_shape, in_shape, in_index)

        in_pos = index_to_position(in_index, in_strides)
        out_pos = index_to_position(out_index, out_strides)

        out[out_pos] = fn(in_storage[in_pos])

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
    ----
        fn: function mappings two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)

        # Compute global thread index
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        # Compute multi-dimensional index for output tensor
        if i >= out_size:
            return

        # Broadcast indices for input tensors
        to_index(i, out_shape, out_index)

        # Compute flat indices for input and output tensors
        broadcast_index(out_index, out_shape, a_shape, a_index)
        broadcast_index(out_index, out_shape, b_shape, b_index)

        # Compute flat indices for input and output tensors
        a_pos = index_to_position(a_index, a_strides)
        b_pos = index_to_position(b_index, b_strides)
        out_pos = index_to_position(out_index, out_strides)

        # Apply binary function to store the result
        out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    """Practice CUDA kernel for summing elements within blocks.

    This kernel performs a block-wise sum of an input array `a` and stores
    the results in the output array `out`. Shared memory is used for efficiency.

    Args:
    ----
        out (Storage): Storage for the output tensor.
        a (Storage): Storage for the input tensor.
        size (int): Length of the input tensor `a`.

    Notes:
    -----
    - The input tensor is divided into blocks of size `BLOCK_DIM`.
    - Each block computes the sum of its elements using shared memory.
    - The sum for each block is written to the corresponding cell in `out`.

    """
    BLOCK_DIM = 32

    # Shared memory allocation
    cache = cuda.shared.array(BLOCK_DIM, numba.float64)

    # Compute global and local thread indices
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = cuda.threadIdx.x

    # Load input data into shared memory
    if i < size:
        cache[pos] = a[i]
    else:
        cache[pos] = 0.0
    cuda.syncthreads()

    # Perform reduction within the block
    step = 1
    while step < cuda.blockDim.x:
        if pos % (2 * step) == 0 and (pos + step) < cuda.blockDim.x:
            cache[pos] += cache[pos + step]
        cuda.syncthreads()
        step *= 2

    # Write the block sum to the output array
    if pos == 0:
        out[cuda.blockIdx.x] = cache[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    """Compute the sum of elements in a tensor using a CUDA kernel.

    This function invokes the `_sum_practice` kernel to compute a block-wise
    sum of the input tensor.

    Args:
    ----
        a (Tensor): The input tensor to be summed.

    Returns:
    -------
        TensorData: A tensor containing the block-wise sums.

    """
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """CUDA higher-order tensor reduce function.

    Args:
    ----
        fn: reduction function maps two floats to float.

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        # BLOCK_DIM = 512
        # cache = cuda.shared.array(BLOCK_DIM, numba.float64)  # Shared memory
        out_index = cuda.local.array(
            MAX_DIMS, numba.int32
        )  # Local array for output index
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x  # Global thread index

        if i < out_size:
            to_index(i, out_shape, out_index)
            out_pos = index_to_position(out_index, out_strides)

            for j in range(a_shape[reduce_dim]):
                out_index[reduce_dim] = j
                in_pos = index_to_position(out_index, a_strides)
                reduce_value = fn(reduce_value, a_storage[in_pos])

            out[out_pos] = reduce_value

    return jit(_reduce)  # Compile the kernel with Numba


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """__mm_practice is a CUDA matrix multiplication kernel.

    This kernel assumes that the input and output matrices are in row-major
    order. The kernel divides the matrix into blocks of size BLOCK_DIM x
    BLOCK_DIM and performs the matrix multiplication block by block.

    The kernel uses shared memory to store the tiles of the input matrices,
    and uses registers to store the partial results of the multiplication.

    The kernel also uses synchronization barriers to ensure that all threads
    in the block have finished loading the data before performing the
    multiplication.

    Parameters
    ----------
    out : Storage
        Storage for the output matrix.
    a : Storage
        Storage for the input matrix A.
    b : Storage
        Storage for the input matrix B.
    size : int
        Size of the matrix (number of rows and columns).

    Returns
    -------
    None

    """
    # Define the block size (# threads per dimension in a block).
    # Each block will process a submatrix (tile) of a result.
    BLOCK_DIM = 32

    # Allocate shared memory to hold tiles for the current block.
    # These are small parts of matrices A nad B that each thread block will process.
    shared_a = cuda.shared.array(
        (BLOCK_DIM, BLOCK_DIM), numba.float32
    )  # Shared memory for tile A
    shared_b = cuda.shared.array(
        (BLOCK_DIM, BLOCK_DIM), numba.float32
    )  # Shared memory for tile B

    # Get the thread index within the block (local position within the block)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    # Get the block index within the grid (global block position within the grid of the block)
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y

    # Calculate global row and column indices for the output matrix
    # These indices will be used to determing which element of the output matrix each thread will compute.
    row = by * cuda.blockDim.y + ty
    col = bx * cuda.blockDim.x + tx

    # Initialize the result accumulator for the result of this thread's computation.
    # This will hold the final value of out[row, col]
    result = 0.0

    # Iterate over the tiles of matrices A and B. Each iteration will process one tile of A and one tile of B.
    # The number of tiles is the ceiling(size/BLOCK_DIM) to ensure we cover the entire block matrix.
    for t in range((size + BLOCK_DIM - 1) // BLOCK_DIM):
        # Load a tile of A into shared memory
        # This is the submatrix of A starting from the current row moving horizontally.
        if row < size and (t * BLOCK_DIM + tx) < size:
            shared_a[ty, tx] = a[row * size + (t * BLOCK_DIM + tx)]
        else:  # Load 0 into shared memory if the indices are out of bounds
            shared_a[ty, tx] = 0.0

        # Load a tile of B into shared memory
        # This is the submatrix of A starting from the current column moving vertically.
        if col < size and (t * BLOCK_DIM + ty) < size:
            shared_b[ty, tx] = b[(t * BLOCK_DIM + ty) * size + col]
        else:  # Load 0 into shared memory if the indices are out of bounds
            shared_b[ty, tx] = 0.0

        # Synchronize threads to ensure all threads have loaded their respective data before any thread continues to process them.
        cuda.syncthreads()

        # Compute the partial result for this thread's element of the output matrix.
        # This invovles multiplying elements of hte current tiles of A and B and summing the results.
        for k in range(BLOCK_DIM):
            # Multiply the k-th element from the current row of A's tile with the k-th element
            # from the current column of B's tile and accumulate the product.
            result += shared_a[ty, k] * shared_b[k, tx]

        # Synchronize threads to ensure all threads of the current tile have completed processing before loading the next tile
        cuda.syncthreads()

    # Write the result to the global output matrix if it is within bounds.
    # This stores he computed value into the correspondign position in the output matrix.
    if row < size and col < size:
        out[row * size + col] = result


jit_mm_practice = jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    """CUDA kernel for matrix multiplication using shared memory.

    This kernel performs block-based matrix multiplication using tiles stored
    in shared memory for efficient computation.

    Args:
    ----
        out (Storage): Storage for the output matrix.
        a (Storage): Storage for the first input matrix.
        b (Storage): Storage for the second input matrix.
        size (int): Number of rows and columns in the square matrices.

    Notes:
    -----
    - Input and output matrices are assumed to be in row-major order.
    - The kernel divides matrices into tiles of size `BLOCK_DIM x BLOCK_DIM`.
    - Synchronization barriers ensure correctness during computation.

    """
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    # These lines calculate the "batch stride" for tensors a and (which determines how to skip
    # between batches). In the case that there is only one batch, the stride is set to 0.
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    # This retrieves the batch index for the current thread block. In CUDA, the grid of threads
    # have 3 dimensions (x, y, and z). The z dimension is here to handle batches.
    batch = cuda.blockIdx.z

    # The size of the block (# of threads in one dimension of thread block (pre-initializzed)
    BLOCK_DIM = 32

    # Shared memory allocated within each block. Shared memory is used because its much
    # faster than global memory, and is used to store submatrices of a and b for faster computation.
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # The global position of the threads (row, column).
    row = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    col = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # Local thread indices within the block. These are used to index within shared memroy.
    thread_x = cuda.threadIdx.x
    thread_y = cuda.threadIdx.y

    # 'value' is the accumulator for the dot product of the corresponding row of a and column of b.
    # Each thread will compute one elemnet of the result matrix.
    value = 0.0

    # This loop moves thorugh the shred dimension (columsn of a and rows of b)  in size BLOCK_DIM.
    # THe loop ensures that all parts of the matrices are processesd, even if the matric's dimension arent
    # exact multiples of BLOCK_DIM.
    for tile in range((a_shape[-1] + BLOCK_DIM - 1) // BLOCK_DIM):
        # Check if thread is within bounds of the matric deimnsions and load the corresponding
        # element from global memory (a_storage) into the shared memory (a_shared).
        # In the case the thread is out of bounds (e.g. when the matrix
        # dimensions do not perfectly align with block size), assign 0.0 to shared memory to pad the tile.
        # (as multiplying by 0 does not affect the final result)
        if row < a_shape[-2] and tile * BLOCK_DIM + thread_y < a_shape[-1]:
            # Calculate the global memory address for element of a:
            # 1. batch * a_batch_stride: batch offset (if batching is used)
            # 2. row * a_strides[-2]: row offset
            # 3. (tile * BLOCK_DIM + thread_y) * a_strides[-1]: column offset
            a_shared[thread_x, thread_y] = a_storage[
                batch * a_batch_stride
                + row * a_strides[-2]
                + (tile * BLOCK_DIM + thread_y) * a_strides[-1]
            ]
        else:
            a_shared[thread_x, thread_y] = 0.0

        # Do the same thing for b (Checking bounds and loading into shared memory)
        if tile * BLOCK_DIM + thread_x < b_shape[-2] and col < b_shape[-1]:
            # Calculate the global memory address for element of b:
            # 1. batch * b_batch_stride: batch offset (if batching is used)
            # 2. (tile * BLOCK_DIM + thread_x) * b_strides[-2]: row offset
            # 3. col * b_strides[-1]: column offset
            b_shared[thread_x, thread_y] = b_storage[
                batch * b_batch_stride
                + (tile * BLOCK_DIM + thread_x) * b_strides[-2]
                + col * b_strides[-1]
            ]
        else:
            b_shared[thread_x, thread_y] = 0.0

        # Synchronize threads to ensure tiles are fully loaded before proceding with computation.
        cuda.syncthreads()

        # Each thread computes part of the dot product for its assigned row-column pair using the
        # shared memroy tiles. The k loop iterates over the elements of the shared_memory tiles
        for k in range(BLOCK_DIM):
            value += a_shared[thread_x, k] * b_shared[k, thread_y]

        # Synchronize threads before loading the next tile
        cuda.syncthreads()

    # Check if the threads' position is within bounds of the output matrix and write the result to the
    # output matrix. Calculate the flattened index (out_pos) for the output tensor using strides. Write the computed
    # value back to global memory.
    if row < out_shape[-2] and col < out_shape[-1]:
        out_pos = batch * out_strides[0] + row * out_strides[1] + col * out_strides[2]
        out[out_pos] = value


tensor_matrix_multiply = jit(_tensor_matrix_multiply)
