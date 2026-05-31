# CUDA-RMM

Optimized implementation of the Reducing Matrix Multiplication (RMM) algorithm using NVIDIA CUDA.

This project was developed for the EPFL Parallelism & Concurrency course and focuses on GPU acceleration techniques such as algebraic reformulation, shared-memory tiling, kernel fusion, and memory-access optimization.

## Overview

Given:

- Matrix A of size M × N
- Matrix B of size N × K

The RMM operation produces:

- Matrix C of size (M/2) × (K/2)

where

\[
C_{i,j}
=
\sum_{a=0}^{1}
\sum_{b=0}^{1}
\sum_{k=0}^{N-1}
A_{2i+a,k}
B_{k,2j+b}
\]

The implementation exploits the equivalent reformulation

\[
(A_{2i,k}+A_{2i+1,k})
(B_{k,2j}+B_{k,2j+1})
\]

which significantly reduces arithmetic work while preserving correctness.

---

## Optimizations

### Algebraic Reformulation

Reduces the number of arithmetic operations by collapsing four products into a single product of row and column reductions.

### GPU-Side Reduction

Moves preprocessing from CPU to GPU to eliminate host-side bottlenecks.

### Shared-Memory Tiling

Uses 16×16 thread blocks and cooperative tile loading to improve data reuse and reduce global-memory traffic.

### Kernel Fusion

Performs reduction and multiplication directly during tile loading, avoiding:

- Intermediate matrices
- Extra global-memory writes
- Additional kernel launches

### Compiler Optimizations

- `__restrict__`
- Loop unrolling
- Coalesced memory accesses

---

## Performance

Measured on NVIDIA GPUs using CUDA event timing.

### Optimization Progression (N = 4096)

| Version | Runtime (s) |
|----------|------------|
| Baseline GPU | 0.1933 |
| CPU Reduction + GPU Multiply | 0.1393 |
| GPU Reduction | 0.1316 |
| Shared-Memory Tiling | 0.1253 |
| Fused Final Kernel | 0.1215 |

Overall speedup relative to the direct GPU baseline:

**1.59×**

### CPU vs GPU

| N | CPU (s) | GPU (s) | Speedup |
|----|---------|---------|----------|
| 256 | 0.0111 | 0.000635 | 17.4× |
| 1024 | 3.375 | 0.00294 | 1146× |
| 4096 | 605.5 | 0.0432 | 14006× |

---

## Build

Compile using:

```bash
make all
```

---

## Run

```bash
./assignment3 M N K 0
```

Example:

```bash
./assignment3 1024 1024 1024 0
```

Output:

```text
Host to Device MemCpy takes ...
RMM operation takes ...
Device to Host MemCpy takes ...
Total time taken ...
```

The resulting matrix is written to:

```text
matC.csv
```

---

## Project Structure

```text
.
├── assignment3.cu
├── rmm.cu
├── utility.h
├── Makefile
├── execute.sh
└── README.md
```

---

## Key CUDA Concepts Demonstrated

- CUDA kernel design
- Thread/block decomposition
- Shared memory
- Synchronization
- Occupancy considerations
- Kernel fusion
- Memory-bandwidth optimization
- GPU performance analysis

---

EPFL — Parallelism & Concurrency
