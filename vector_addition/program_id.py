# %%
import torch
import triton
import triton.language as tl

# %%
@triton.jit
def test_kernel():
    pid = tl.program_id(axis=0)
    print(f"Program ID: ", pid)

# %%
grid = (2, )
test_kernel[grid]()  # Launch 2 programs
# note that each pid has 32 prints -> 32 threads per warp
# GPU smallest unit of execution is a warp

# if you don't want to print 32 lines...
# %%
@triton.jit
def test_kernel(output_ptr):
    pid = tl.program_id(axis=0)
    tl.store(output_ptr + pid, pid)

n_elements = 4
output = torch.zeros(n_elements, dtype=torch.int32, device="cuda")
test_kernel[(n_elements,)](output)
print(output.cpu()) 
