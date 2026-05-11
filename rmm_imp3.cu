/*
============================================================================
Filename    : rmm_imp3.cu
Author      : Guillaume Lepin & Neha Chakraborty
SCIPER      : 381189 & 373384
============================================================================
Optimization: reduce A row-pairs and B column-pairs on the GPU, then run a shared-memory tiled
*/

#include <iostream>
#include <iomanip>
#include <sys/time.h>
#include <cuda_runtime.h>
using namespace std;

#define TILE_SIZE 16

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

__global__ void rmm_kernel(const int *__restrict__ ptrA,
                           const int *__restrict__ ptrB,
                           int *__restrict__ ptrC,
                           int M, int N, int K)
{
    __shared__ int tileA[TILE_SIZE][TILE_SIZE];
    __shared__ int tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    int outRows = M / 2;
    int outCols = K / 2;
    int sum = 0;

    for(int tileStart = 0; tileStart < N; tileStart += TILE_SIZE) {
        int kA = tileStart + threadIdx.x;
        int kB = tileStart + threadIdx.y;

        if(row < outRows && kA < N) {
            tileA[threadIdx.y][threadIdx.x] =
                ptrA[(2 * row) * N + kA] + ptrA[(2 * row + 1) * N + kA];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0;
        }

        if(kB < N && col < outCols) {
            tileB[threadIdx.y][threadIdx.x] =
                ptrB[kB * K + 2 * col] + ptrB[kB * K + 2 * col + 1];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        #pragma unroll
        for(int kdx = 0; kdx < TILE_SIZE; kdx++) {
            sum += tileA[threadIdx.y][kdx] * tileB[kdx][threadIdx.x];
        }

        __syncthreads();
    }

    if(row < outRows && col < outCols) {
        ptrC[row * outCols + col] = sum;
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

    int outRows = M / 2;
    int outCols = K / 2;
    size_t sizeA = (size_t) M * N * sizeof(int);
    size_t sizeB = (size_t) N * K * sizeof(int);
    size_t sizeC = (size_t) outRows * outCols * sizeof(int);

    bool ok = true;
    ok = ok && check_cuda(cudaMalloc((void **) &ptrA, sizeA), "Error allocating memory for matA on device");
    ok = ok && check_cuda(cudaMalloc((void **) &ptrB, sizeB), "Error allocating memory for matB on device");
    ok = ok && check_cuda(cudaMalloc((void **) &ptrC, sizeC), "Error allocating memory for matC on device");
    if(!ok) {
        cudaFree(ptrA);
        cudaFree(ptrB);
        cudaFree(ptrC);
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
        return;
    }

    cudaEventRecord(comp_start);

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((outCols + TILE_SIZE - 1) / TILE_SIZE,
                 (outRows + TILE_SIZE - 1) / TILE_SIZE);
    rmm_kernel<<<gridDim, blockDim>>>(ptrA, ptrB, ptrC, M, N, K);

    ok = check_cuda(cudaGetLastError(), "Error launching RMM kernel");
    cudaEventRecord(comp_end);
    ok = ok && check_cuda(cudaEventSynchronize(comp_end), "Error executing RMM kernel");
    if(!ok) {
        cudaFree(ptrA);
        cudaFree(ptrB);
        cudaFree(ptrC);
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
