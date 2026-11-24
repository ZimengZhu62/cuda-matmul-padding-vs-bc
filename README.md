# cuda-matmul-padding-vs-bc
A small experiment comparing two approaches to implementing matrix multiplication

CUDA Matrix Multiplication: Padding vs. Boundary Checking

1. Algorithm Design

This experiment compares two approaches to handling matrix boundaries in CUDA matrix multiplication.

Both implementations use a 16 × 16 shared-memory tiled matrix multiplication kernel.

Padding

Before GPU execution, the input matrices are padded to dimensions divisible by 16.

This allows the GPU kernel to perform regular tiled computation without boundary checks. After computation, the output matrix is cropped to its original dimensions.

Boundary Checking (BC)

Instead of padding the matrices, the original dimensions are preserved.

The GPU kernel handles boundary conditions directly by checking whether loaded elements are within valid matrix boundaries. Invalid elements are replaced with zero.

The goal is to investigate whether avoiding boundary checks through padding actually improves overall performance.

2. Experiment

Experiment design: Proposed and designed by me. The benchmark code and CUDA implementation assistance were provided by ChatGPT.

The experiment compares both implementations using identical input matrices and a CPU reference implementation for correctness verification.

Configuration	Value
GPU	NVIDIA RTX 5060
CUDA	13.5
Tile Size	16 × 16
Warmup	3 iterations
Repetitions	20 iterations
GPU Timing	CUDA Events
CPU Timing	QueryPerformanceCounter

Ten different matrix shapes were tested, including square and non-square matrices.

End-to-End Results

Matrix Dimensions	Padding (ms)	BC (ms)	BC Speedup
2 × 2 × 2	         0.0984	       0.0962	  1.02×
64 × 64 × 64	     0.1429	       0.1159	  1.23×
256 × 256 × 256	   0.5030	       0.2755	  1.83×
512 × 512 × 512	   1.5715	       0.7392 	2.13×
1024 × 1024 × 1024 6.2771	       3.0431	  2.06×
64 × 1024 × 64	   0.3773	       0.2001  	1.89×
1024 × 64 × 64	   0.5084	       0.3240	  1.57×
128 × 128 × 2048	 0.9724	       0.4570	  2.13×
256 × 2048 × 128	 1.5534	       0.5761	  2.70×
1000 × 1000 × 1000 6.0575	       2.9401	  2.06×

All test cases passed CPU reference verification.

Kernel-Level Comparison

The GPU kernel execution times were also measured separately.

The results showed that neither approach consistently outperformed the other at the kernel level.

For example, with a 1024 × 1024 × 1024 matrix:

* Padding: 1.4715 ms
* BC: 1.4701 ms

The two kernels performed almost identically.

3. Conclusion

Although Padding eliminates boundary checks inside the GPU kernel, it does not provide a consistent kernel-level performance advantage.

Its additional CPU preprocessing and output cropping introduce considerable overhead.

In the 1024³ test, CPU padding and cropping alone took approximately 3.28 ms, while the GPU kernel took only around 1.47 ms.

As a result, BC achieved better end-to-end performance in all ten test cases, with speedups ranging from 1.02× to 2.70×.

The main takeaway: A simpler GPU kernel does not necessarily lead to a faster overall program. The cost of preprocessing and memory operations must also be considered.

This experiment compares the two specific implementations and does not establish a universal performance advantage for either approach.
