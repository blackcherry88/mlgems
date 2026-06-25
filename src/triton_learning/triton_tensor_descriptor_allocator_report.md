# Report: do we need `triton.set_allocator(...)` for `triton_2d_block_tensor_desc.py`?

Date: 2026-06-23  
Local Triton version checked: `3.7.1`

## Bottom line

Yes. Keep this in `triton_2d_block_tensor_desc.py`:

```python
def triton_alloc_callback(size: int, alignment: int, stream: Optional[int]):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(triton_alloc_callback)
```

For this specific tensor-descriptor kernel, the allocator is required on NVIDIA GPUs with TMA support, especially Hopper/Blackwell-class GPUs such as SM90 and SM100. Without it, the kernel can compile, but launch can fail when Triton asks for runtime global scratch memory for the tensor descriptors.

On pre-Hopper NVIDIA GPUs such as SM80, this exact kernel does not need the allocator because Triton rewrites tensor descriptors into normal pointer-style memory operations. Even there, keeping the allocator is harmless and makes the script portable across newer GPUs.

## File under discussion

[`triton_2d_block_tensor_desc.py`](./triton_2d_block_tensor_desc.py) creates two tensor descriptors inside the JIT kernel:

```python
x_desc = tl.make_tensor_descriptor(...)
o_desc = tl.make_tensor_descriptor(...)
```

The launch shape is:

```python
M, N = 130, 200
BLOCK_M, BLOCK_N = 64, 64
grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
```

So the grid is:

```text
ceil(130 / 64) x ceil(200 / 64) = 3 x 4 = 12 programs
```

## Why an allocator is involved

`tl.make_tensor_descriptor` is not just syntactic sugar on Hopper/Blackwell GPUs. On NVIDIA GPUs with TMA support, Triton lowers the descriptor to a TMA-backed descriptor object. TMA descriptors need runtime global memory workspace.

Triton’s runtime allocator hook is exactly for this class of problem. In the installed Triton source, `triton.set_allocator(...)` registers a callback that is called at kernel launch for kernels requiring additional global workspace.

If no allocator is registered, Triton uses a null allocator. When a launch needs scratch memory, that null allocator raises an error telling the user to call `triton.set_allocator`.

## Architecture-dependent behavior

The behavior is different across NVIDIA architecture generations.

| Target | Compiler behavior | Scratch per program | TMA used? | Is `set_allocator` needed for this kernel? |
|---|---:|---:|---:|---:|
| SM80 / Ampere | Tensor descriptors rewritten to pointer operations | `0` bytes | No | Not strictly |
| SM90 / Hopper | Tensor descriptors lowered to TMA | `256` bytes | Yes | Yes |
| SM100 / Blackwell | Tensor descriptors lowered to TMA | `256` bytes | Yes | Yes |

The local compile-only check produced:

```text
80 scratch_size= 0 scratch_align= 1 has_tma= False
90 scratch_size= 256 scratch_align= 128 has_tma= True
100 scratch_size= 256 scratch_align= 128 has_tma= True
```

For this file, the SM90/SM100 scratch calculation is:

```text
global_scratch_size per program = 256 bytes
grid size                       = 3 * 4 * 1 = 12 programs
num_ctas                        = 1
runtime allocation              = 256 * 12 * 1 = 3072 bytes
alignment                       = 128 bytes
```

That 256 bytes matches the two descriptors in the kernel: one input descriptor and one output descriptor.

## Reproduction command

The compile-only helper in this folder can reproduce the metadata check without launching on a GPU:

```bash
env TRITON_CACHE_DIR=/tmp/triton-cache python src/triton_learning/compile_triton_descriptor.py
```

Expected output on the checked environment:

```text
80 scratch_size= 0 scratch_align= 1 has_tma= False
90 scratch_size= 256 scratch_align= 128 has_tma= True
100 scratch_size= 256 scratch_align= 128 has_tma= True
```

Note: the current local environment had `torch.cuda.is_available() == False`, so the evidence above is from Triton compilation metadata, not a live CUDA launch.

## What happens if we remove it?

If the allocator is removed:

- On SM80/pre-Hopper, this particular kernel should still launch because no global scratch allocation is requested.
- On SM90/SM100, the runtime will request descriptor scratch memory. With no allocator installed, launch is expected to fail.

So removing the allocator would make the example accidentally architecture-dependent: okay on Ampere, broken on Hopper/Blackwell.

## Is the current callback good enough?

Yes. This is the same style as Triton’s own documented example:

```python
def triton_alloc_callback(size: int, alignment: int, stream: Optional[int]):
    return torch.empty(size, device="cuda", dtype=torch.int8)
```

The callback only needs to return a CUDA allocation object with a usable `data_ptr()`. A `torch.empty(..., dtype=torch.int8, device="cuda")` tensor is suitable because it creates a raw byte buffer on the GPU. The `alignment` argument is passed by Triton; PyTorch’s CUDA allocator is already sufficiently aligned for this use in normal practice, and Triton’s own example uses this pattern.

Register it once before the first launch:

```python
triton.set_allocator(triton_alloc_callback)
```

It does not need to be called before every kernel launch.

## Recommended decision for this repo

Keep the allocator registration in `triton_2d_block_tensor_desc.py`.

Reason:

- The script uses `tl.make_tensor_descriptor`.
- The kernel has two descriptors.
- On SM90+, Triton lowers those descriptors to TMA.
- TMA descriptor lowering requires runtime global scratch memory.
- The current allocator callback is simple, official-style, and harmless on GPUs where no scratch is needed.

If this file is meant as a learning example, keeping the allocator is also educational: it shows the real host-side requirement that comes with tensor descriptors/TMA on modern NVIDIA GPUs.

## Sources checked

Local installed Triton source:

- `/home/taohu/.pyenv/versions/mlgems/lib/python3.12/site-packages/triton/language/core.py`
  - `tl.make_tensor_descriptor` notes that NVIDIA GPUs with TMA support use TMA-backed descriptors.
  - The example registers an allocator before launching a descriptor kernel.
- `/home/taohu/.pyenv/versions/mlgems/lib/python3.12/site-packages/triton/runtime/_allocation.py`
  - `set_allocator` registers the launch-time workspace allocator.
  - `NullAllocator` raises if a kernel requires allocation and no allocator is set.
- `/home/taohu/.pyenv/versions/mlgems/lib/python3.12/site-packages/triton/backends/nvidia/compiler.py`
  - For capability `< 90`, Triton rewrites tensor descriptors to pointer operations.
  - For capability `>= 90`, Triton runs TMA lowering.
- `/home/taohu/.pyenv/versions/mlgems/lib/python3.12/site-packages/triton/backends/nvidia/driver.py`
  - Runtime allocation size is computed from grid size, `num_ctas`, and compiled global scratch size.

Upstream references:

- Triton `make_tensor_descriptor` API docs: <https://triton-lang.org/main/python-api/generated/triton.language.make_tensor_descriptor.html>
- Triton v3.7.1 `core.py`: <https://github.com/triton-lang/triton/blob/v3.7.1/python/triton/language/core.py>
- Triton v3.7.1 runtime allocation source: <https://github.com/triton-lang/triton/blob/v3.7.1/python/triton/runtime/_allocation.py>
- Triton v3.7.1 NVIDIA compiler source: <https://github.com/triton-lang/triton/blob/v3.7.1/third_party/nvidia/backend/compiler.py>
- Triton v3.7.1 NVIDIA driver source: <https://github.com/triton-lang/triton/blob/v3.7.1/third_party/nvidia/backend/driver.py>
- Triton tensor descriptor tests: <https://github.com/triton-lang/triton/blob/main/python/test/unit/language/test_tensor_descriptor.py>
