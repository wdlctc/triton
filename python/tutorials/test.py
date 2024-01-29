
# %%
# Compute Kernel
# --------------

import torch

import triton
import triton.language as tl


@triton.jit
def matmul_kernel(x_ptr,  # *Pointer* to first input vector.
                  y_ptr,  # *Pointer* to second input vector.
                  output_ptr,  # *Pointer* to output vector.
                  n_elements,  # Size of the vector.
                  stride, 
                  BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
                  # NOTE: `constexpr` so it can be used as a shape value.
                  ):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    
    col_offsets = tl.arange(0, BLOCK_SIZE)


    x_start_ptr = x_ptr
    x_ptrs = x_start_ptr + col_offsets

    y_start_ptr = y_ptr
    y_ptrs = y_start_ptr + col_offsets
    
    # block_start = pid * BLOCK_SIZE
    # offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # mask = offsets < n_elements
    # mask_y = offsets[None, :] < n_elements

    output = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    x = tl.load( x_start_ptr + col_offsets, mask=col_offsets < n_elements)
    for k in range(0, n_elements):
        y = tl.load( y_start_ptr + col_offsets, mask=col_offsets < n_elements)
        output = x * y
        output_sum = tl.sum(output)
        tl.store(output_ptr, output_sum)
        y_start_ptr += n_elements
        output_ptr += 1


def matmul(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    matmul_kernel[grid](x, y, output, n_elements, x.stride(0), BLOCK_SIZE=16384)
    return output


torch.manual_seed(0)
size = 1024
x = torch.rand((1, size), device='cuda')
y = torch.rand((size, size), device='cuda').contiguous()

output_triton = matmul(x, y)
output_torch = torch.nn.functional.linear(x, y, bias=None)
print(output_torch)
print(output_triton)
print(x,y)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')



@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[16384],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton', 'torch'],  # Possible values for `line_arg`.
        line_names=['Triton', 'Torch'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='vector-add-performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark(size, provider):
    x = torch.rand((1, size), device='cuda', dtype=torch.float32)
    y = torch.rand((size, size), device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.nn.functional.linear(x, y, bias=None), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(x, y), quantiles=quantiles)
    gbps = lambda ms: 12 * size / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)


# %%
# We can now run the decorated function above. Pass `print_data=True` to see the performance number, `show_plots=True` to plot them, and/or
# `save_path='/path/to/results/' to save them to disk along with raw CSV data:
benchmark.run(print_data=True, show_plots=True)
