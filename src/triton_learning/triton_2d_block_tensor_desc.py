import torch
import triton
import triton.language as tl
from typing import Optional

# --- 1. Register Allocator ---
def triton_alloc_callback(size: int, alignment: int, stream: Optional[int]):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(triton_alloc_callback)

# --- 2. Kernel Definition ---
@triton.jit
def load_2d_tensor_desc_kernel(
    x_ptr, o_ptr, M, N, stride_xm, stride_xn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Declare Stateless Descriptors
    x_desc = tl.make_tensor_descriptor(
        base=x_ptr, shape=[M, N], strides=[stride_xm, stride_xn],
        block_shape=[BLOCK_M, BLOCK_N], padding_option='zero'
    )
    o_desc = tl.make_tensor_descriptor(
        base=o_ptr, shape=[M, N], strides=[stride_xm, stride_xn],
        block_shape=[BLOCK_M, BLOCK_N], padding_option='zero'
    )
    
    # Query Coordinates
    start_m = pid_m * BLOCK_M
    start_n = pid_n * BLOCK_N
    
    # Perform I/O
    block = x_desc.load([start_m, start_n])
    o_desc.store([start_m, start_n], block)

# --- 3. Host Driver ---
def test_tensor_descriptor():
    device = "cuda"
    M, N = 130, 200  # Non-power-of-two to test boundary clamping
    BLOCK_M, BLOCK_N = 64, 64
    
    # Initialize data
    x = torch.randn((M, N), device=device, dtype=torch.float32)
    out = torch.zeros_like(x)
    
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']), 
        triton.cdiv(N, meta['BLOCK_N'])
    )
    
    # Launch Kernel
    load_2d_tensor_desc_kernel[grid](
        x, out, M, N, x.stride(0), x.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
    )
    
    # Verify results
    success = torch.allclose(x, out)
    print(f"Tensor Descriptor Kernel Execution Successful: {success} 🎉")

test_tensor_descriptor()