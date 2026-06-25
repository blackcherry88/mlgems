import torch
import triton
import triton.language as tl

# =====================================================================
# 1. Triton Kernel Definitions
# =====================================================================

@triton.jit
def load_2d_broadcast_kernel(
    x_ptr, o_ptr, M, N, stride_xm, stride_xn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    start_m = pid_m * BLOCK_M
    start_n = pid_n * BLOCK_N
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = start_n + tl.arange(0, BLOCK_N)

    # if pid_m == 0 and pid_n == 0:
    #     print(f"offs_m values: {offs_m}")
    #     print(f"offs_n values: {offs_n}")
    
    # Broadcast to 2D grid pointer array
    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    # Load and Store with manual masks
    block = tl.load(x_ptrs, mask=mask, other=0.0)
    
    o_ptrs = o_ptr + (offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn)
    tl.store(o_ptrs, block, mask=mask)


@triton.jit
def load_2d_block_ptr_kernel(
    x_ptr, o_ptr, M, N, stride_xm, stride_xn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Define Block Pointer
    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(M, N),
        strides=(stride_xm, stride_xn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0) 
    )
    
    # Load using block pointer hardware constraints
    block = tl.load(x_block_ptr, boundary_check=(0, 1))
    
    o_block_ptr = tl.make_block_ptr(
        base=o_ptr,
        shape=(M, N),
        strides=(stride_xm, stride_xn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0)
    )
    tl.store(o_block_ptr, block, boundary_check=(0, 1))


@triton.jit
def load_2d_tensor_desc_kernel(
    x_ptr, o_ptr, M, N, stride_xm, stride_xn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # 1. Create a global descriptor of the entire tensor structure
    # 'padding_option' automatically handles boundary out-of-bound masking with 0s!
    x_desc = tl.make_tensor_descriptor(
        base=x_ptr,
        shape=[M, N],
        strides=[stride_xm, stride_xn],
        block_shape=[BLOCK_M, BLOCK_N],
        padding_option='zero' 
    )
    
    o_desc = tl.make_tensor_descriptor(
        base=o_ptr,
        shape=[M, N],
        strides=[stride_xm, stride_xn],
        block_shape=[BLOCK_M, BLOCK_N],
        padding_option='zero'
    )
    
    # 2. Compute dynamic start coordinates for this specific program block
    start_m = pid_m * BLOCK_M
    start_n = pid_n * BLOCK_N
    
    # 3. Load and Store directly using the descriptor methods
    block = x_desc.load([start_m, start_n])
    
    o_desc.store([start_m, start_n], block)

# =====================================================================
# 2. Host Driving & Verification Program
# =====================================================================

def test_kernels():
    # Ensure CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is required to run Triton kernels.")
        return

    device = "cuda"
    
    # Non-power-of-two matrix sizes to rigorously test boundary conditions
    M, N = 130, 200  
    BLOCK_M, BLOCK_N = 64, 64
    
    print(f"Matrix Size: {M}x{N}")
    print(f"Block Size:  {BLOCK_M}x{BLOCK_N}")
    print("-" * 40)

    # Initialize a random input matrix
    x = torch.randn((M, N), device=device, dtype=torch.float32)
    
    # Allocate separate output matrices filled with zeros
    out_broadcast = torch.zeros_like(x)
    out_block_ptr = torch.zeros_like(x)
    
    # Get strides 
    stride_xm, stride_xn = x.stride(0), x.stride(1)
    print(f"Input Stride (M, N): ({stride_xm}, {stride_xn})")
    
    # Define 2D Grid launch configuration
    # triton.cdiv performs ceiling division (e.g., cdiv(130, 64) = 3 blocks)
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']), 
        triton.cdiv(N, meta['BLOCK_N'])
    )
    
    # --- Launch Method A (Broadcast) ---
    load_2d_broadcast_kernel[grid](
        x, out_broadcast, M, N, stride_xm, stride_xn,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
    )
    
    # --- Launch Method B (Block Pointer) ---
    load_2d_block_ptr_kernel[grid](
        x, out_block_ptr, M, N, stride_xm, stride_xn,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
    )
    
    # --- Verification ---
    # Check if the output elements match the input elements
    broadcast_correct = torch.allclose(x, out_broadcast)
    block_ptr_correct = torch.allclose(x, out_block_ptr)
    
    # Check if both execution results match each other perfectly
    kernels_match = torch.equal(out_broadcast, out_block_ptr)
    
    print(f"Broadcast Kernel matches original?  {broadcast_correct}")
    print(f"Block Pointer Kernel matches original? {block_ptr_correct}")
    print(f"Do both Triton methods yield identical results? {kernels_match}")
    
    if kernels_match and broadcast_correct:
        print("\nVerification SUCCESS! 🎉")
    else:
        print("\nVerification FAILED. ❌")

if __name__ == "__main__":
    test_kernels()