/*
============================================================================
Filename    : rmm.cu
Author      : Guillaume Lepin & Neha Chakraborty
SCIPER      : 381189 & 373384
============================================================================
*/

#include <iostream>
#include <iomanip>
#include <sys/time.h>
#include <cuda_runtime.h>
using namespace std;

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

/* GPU Optimized Function */
void rmm_gpu(int *matA, int *matB, int *matC, int M, int N, int K)
{
    /* Cuda events for calculating elapsed time */
    cudaEvent_t cpy_H2D_start, cpy_H2D_end, comp_start, comp_end, cpy_D2H_start, cpy_D2H_end;
    cudaEventCreate(&cpy_H2D_start);
    cudaEventCreate(&cpy_H2D_end);
    cudaEventCreate(&comp_start);
    cudaEventCreate(&comp_end);
    cudaEventCreate(&cpy_D2H_start);
    cudaEventCreate(&cpy_D2H_end);

    /* Preprocessing (if any) goes here */

    void *ptrA = nullptr;
    cudaError_t gpu_matA = cudaMalloc(&ptrA, M * N * sizeof(int));
    if(gpu_matA != cudaSuccess) {
        cout << "Error allocating memory for matA on device: " << cudaGetErrorString(gpu_matA) << endl;
        return;
    }

    void *ptrB = nullptr;
    cudaError_t gpu_matB = cudaMalloc(&ptrB, N * K * sizeof(int));
    if(gpu_matB != cudaSuccess) {
        cout << "Error allocating memory for matB on device: " << cudaGetErrorString(gpu_matB) << endl;
        return;
    }

    void *ptrC = nullptr;
    cudaError_t gpu_matC = cudaMalloc(&ptrC, (M/2) * (K/2) * sizeof(int));
    if(gpu_matC != cudaSuccess) {
        cout << "Error allocating memory for matC on device: " << cudaGetErrorString(gpu_matC) << endl;
        return;
    }


    cudaEventRecord(cpy_H2D_start);
    /* Copying array(s) from host to device goes here */

    cudaError_t copyA = cudaMemcpy(ptrA, matA, M * N * sizeof(int), cudaMemcpyHostToDevice);
    if(copyA != cudaSuccess) {
        cout << "Error copying matA from host to device: " << cudaGetErrorString(copyA) << endl;
        return;
    }

    cudaError_t copyB = cudaMemcpy(ptrB, matB, N * K * sizeof(int), cudaMemcpyHostToDevice);
    if(copyB != cudaSuccess) {
        cout << "Error copying matB from host to device: " << cudaGetErrorString(copyB) << endl;
        return;
    }
    

    cudaEventRecord(cpy_H2D_end);
    cudaEventSynchronize(cpy_H2D_end);

    cudaEventRecord(comp_start);
    /* Launching the GPU kernel to do the computation goes here */
    cudaEventRecord(comp_end);
    cudaEventSynchronize(comp_end);

    cudaEventRecord(cpy_D2H_start);
    /* Copying array(s) from device to host goes here */
    cudaEventRecord(cpy_D2H_end);
    cudaEventSynchronize(cpy_D2H_end);

    /* Postprocessing (if any) goes here */

    /* Display timing statistics */
    float time;
    cudaEventElapsedTime(&time, cpy_H2D_start, cpy_H2D_end);
    cout << "Host to Device MemCpy takes " << setprecision(4) << time/1000 << "s" << endl;

    cudaEventElapsedTime(&time, comp_start, comp_end);
    cout << "RMM operation takes " << setprecision(4) << time/1000 << "s" << endl;

    cudaEventElapsedTime(&time, cpy_D2H_start, cpy_D2H_end);
    cout << "Device to Host MemCpy takes " << setprecision(4) << time/1000 << "s" << endl;
}