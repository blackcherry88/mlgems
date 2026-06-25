import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


@triton.jit
def descriptor_kernel(
    x_ptr,
    o_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    x_desc = tl.make_tensor_descriptor(
        base=x_ptr,
        shape=[M, N],
        strides=[stride_xm, stride_xn],
        block_shape=[BLOCK_M, BLOCK_N],
        padding_option="zero",
    )
    o_desc = tl.make_tensor_descriptor(
        base=o_ptr,
        shape=[M, N],
        strides=[stride_xm, stride_xn],
        block_shape=[BLOCK_M, BLOCK_N],
        padding_option="zero",
    )
    start_m = tl.program_id(0) * BLOCK_M
    start_n = tl.program_id(1) * BLOCK_N
    block = x_desc.load([start_m, start_n])
    o_desc.store([start_m, start_n], block)


signature = {
    "x_ptr": "*fp32",
    "o_ptr": "*fp32",
    "M": "i32",
    "N": "i32",
    "stride_xm": "i64",
    "stride_xn": "i64",
    "BLOCK_M": "constexpr",
    "BLOCK_N": "constexpr",
}
source = ASTSource(
    descriptor_kernel,
    signature,
    constexprs={"stride_xn": 1, "BLOCK_M": 64, "BLOCK_N": 64},
)

for capability in (80, 90, 100):
    compiled = triton.compile(
        source,
        target=GPUTarget("cuda", capability, 32),
        options={"num_warps": 4},
    )
    print(
        capability,
        "scratch_size=",
        compiled.metadata.global_scratch_size,
        "scratch_align=",
        compiled.metadata.global_scratch_align,
        "has_tma=",
        "cp.async.bulk.tensor" in compiled.asm["ptx"],
    )
