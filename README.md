# Triton Experiments

These are a collection of Triton kernels from [Triton
Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html). I
plan to slowly write more custom kernels over time.

More interesting are the kernels under `matrix_multiplication`.
`block_scaler.py` seeks to implement a simple fp8 block scaled matrix
multiplication demo. Although it works (with acceptable error), it does not
unlock proper fp8 performance on Ada generation hardware.