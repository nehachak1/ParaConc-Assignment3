/*
============================================================================
Filename    : rmm_imp2.cu
Author      : Guillaume Lepin & Neha Chakraborty
SCIPER      : 381189 & 373384
============================================================================
Optimization: Reduces A/B on the GPU first (not the CPU), then uses a GPU multiply
*/

#include <iostream>
#include <iomanip>
#include <sys/time.h>
#include <cuda_runtime.h>
using namespace std;

#define BLOCK_SIZE 256

static bool check_cuda(cudaError_t err, const char *message)
{
    if(err != cudaSuccess) {
        cout << message << ": " << cudaGetErrorString(err) << endl;
        return false;
    }
    return true;
}

/* CPU Baseline */
void rmm_cpu(int *matA, int *matB, int *matC, int M, int N, int K)
{
    for(int idx = 0; idx < M/2; idx++) {
        for(int jdx = 0; jdx < K/2; jdx++) {
            matC[idx*(K/2) + jdx] = 0;
            for(int aoff = 0; aoff < 2; aoff++) {
                for(int boff = 0; boff < 2; boff++) {
                    for(int kdx = 0; kdx < N; kdx++) {
                        matC[idx*(K/2) + jdx] += matA[(idx*2 + aoff)*N + kdx] * matB[kdx*K + jdx*2 + boff];
                    }
                }
            }
        }
    }
}

__global__ void reduceA_kernel(const int *matA, int *redA, int M, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = (M / 2) * N;

    if(idx < total) {
        int row = idx / N;
        int col = idx % N;
        redA[idx] = matA[(2 * row) * N + col] + matA[(2 * row + 1) * N + col];
    }
}

__global__ void reduceB_kernel(const int *matB, int *redB, int N, int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int outCols = K / 2;
    int total = N * outCols;

    if(idx < total) {
        int row = idx / outCols;
        int col = idx % outCols;
        redB[idx] = matB[row * K + 2 * col] + matB[row * K + 2 * col + 1];
    }
}

__global__ void rmm_kernel(const int *redA, const int *redB, int *matC, int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int outRows = M / 2;
    int outCols = K / 2;

    if(row < outRows && col < outCols) {
        int sum = 0;
        for(int kdx = 0; kdx < N; kdx++) {
            sum += redA[row * N + kdx] * redB[kdx * outCols + col];
        }
        matC[row * outCols + col] = sum;
    }
}

/* GPU Optimized Function */
void rmm_gpu(int *matA, int *matB, int *matC, int M, int N, int K)
{
    cudaEvent_t cpy_H2D_start, cpy_H2D_end, comp_start, comp_end, cpy_D2H_start, cpy_D2H_end;
    cudaEventCreate(&cpy_H2D_start);
    cudaEventCreate(&cpy_H2D_end);
    cudaEventCreate(&comp_start);
    cudaEventCreate(&comp_end);
    cudaEventCreate(&cpy_D2H_start);
    cudaEventCreate(&cpy_D2H_end);

    int *ptrA = nullptr;
    int *ptrB = nullptr;
    int *ptrC = nullptr;
    int *redA = nullptr;
    int *redB = nullptr;

    int outRows = M / 2;
    int outCols = K / 2;
    size_t sizeA = (size_t) M * N * sizeof(int);
    size_t sizeB = (size_t) N * K * sizeof(int);
    size_t sizeC = (size_t) outRows * outCols * sizeof(int);
    size_t sizeRedA = (size_t) outRows * N * sizeof(int);
    size_t sizeRedB = (size_t) N * outCols * sizeof(int);

    bool ok = true;
    ok = ok && check_cuda(cudaMalloc((void **) &ptrA, sizeA), "Error allocating memory for matA on device");
    ok = ok && check_cuda(cudaMalloc((void **) &ptrB, sizeB), "Error allocating memory for matB on device");
    ok = ok && check_cuda(cudaMalloc((void **) &ptrC, sizeC), "Error allocating memory for matC on device");
    ok = ok && check_cuda(cudaMalloc((void **) &redA, sizeRedA), "Error allocating memory for reduced matA on device");
    ok = ok && check_cuda(cudaMalloc((void **) &redB, sizeRedB), "Error allocating memory for reduced matB on device");
    if(!ok) {
        cudaFree(ptrA);
        cudaFree(ptrB);
        cudaFree(ptrC);
        cudaFree(redA);
        cudaFree(redB);
        return;
    }

    cudaEventRecord(cpy_H2D_start);
    ok = ok && check_cuda(cudaMemcpy(ptrA, matA, sizeA, cudaMemcpyHostToDevice),
                          "Error copying matA from host to device");
    ok = ok && check_cuda(cudaMemcpy(ptrB, matB, sizeB, cudaMemcpyHostToDevice),
                          "Error copying matB from host to device");
    cudaEventRecord(cpy_H2D_end);
    ok = ok && check_cuda(cudaEventSynchronize(cpy_H2D_end),
                          "Error synchronizing host to device copy");
    if(!ok) {
        cudaFree(ptrA);
        cudaFree(ptrB);
        cudaFree(ptrC);
        cudaFree(redA);
        cudaFree(redB);
        return;
    }

    cudaEventRecord(comp_start);

    int redABlocks = (outRows * N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int redBBlocks = (N * outCols + BLOCK_SIZE - 1) / BLOCK_SIZE;
    reduceA_kernel<<<redABlocks, BLOCK_SIZE>>>(ptrA, redA, M, N);
    reduceB_kernel<<<redBBlocks, BLOCK_SIZE>>>(ptrB, redB, N, K);

    dim3 blockDim(16, 16);
    dim3 gridDim((outCols + blockDim.x - 1) / blockDim.x,
                 (outRows + blockDim.y - 1) / blockDim.y);
    rmm_kernel<<<gridDim, blockDim>>>(redA, redB, ptrC, M, N, K);

    ok = check_cuda(cudaGetLastError(), "Error launching RMM kernels");
    cudaEventRecord(comp_end);
    ok = ok && check_cuda(cudaEventSynchronize(comp_end), "Error executing RMM kernels");
    if(!ok) {
        cudaFree(ptrA);
        cudaFree(ptrB);
        cudaFree(ptrC);
        cudaFree(redA);
        cudaFree(redB);
        return;
    }

    cudaEventRecord(cpy_D2H_start);
    ok = check_cuda(cudaMemcpy(matC, ptrC, sizeC, cudaMemcpyDeviceToHost),
                    "Error copying matC from device to host");
    cudaEventRecord(cpy_D2H_end);
    ok = ok && check_cuda(cudaEventSynchronize(cpy_D2H_end),
                          "Error synchronizing device to host copy");

    cudaFree(ptrA);
    cudaFree(ptrB);
    cudaFree(ptrC);
    cudaFree(redA);
    cudaFree(redB);

    if(!ok) {
        return;
    }

    float time;
    cudaEventElapsedTime(&time, cpy_H2D_start, cpy_H2D_end);
    cout << "Host to Device MemCpy takes " << setprecision(4) << time/1000 << "s" << endl;

    cudaEventElapsedTime(&time, comp_start, comp_end);
    cout << "RMM operation takes " << setprecision(4) << time/1000 << "s" << endl;

    cudaEventElapsedTime(&time, cpy_D2H_start, cpy_D2H_end);
    cout << "Device to Host MemCpy takes " << setprecision(4) << time/1000 << "s" << endl;
}
