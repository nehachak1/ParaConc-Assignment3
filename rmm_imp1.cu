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

__global__ void rmm_kernel(int *ptrA, int *ptrB, int *ptrC, int M, int N, int K) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if(row < M/2 && col < K/2) {
            int sum = 0;
            for(int kdx = 0; kdx < N; kdx++) {
                sum += ptrA[row*N + kdx] * ptrB[kdx*(K/2) + col];
            }
            ptrC[row*(K/2) + col] = sum;
        }
        // Kernel implementation goes here
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

    int *reducedA = (int *) malloc((M/2) * N * sizeof(int));
    int *reducedB = (int *) malloc(N * (K/2) * sizeof(int));

    for(int i = 0; i < M/2; i++){
        for(int j = 0; j < N; j++){
            reducedA[i*N + j] = matA[(i*2)*N + j] + matA[(i*2 + 1)*N + j];
        }
    }

    for(int i = 0; i < N; i++){
        for(int j = 0; j < K/2; j++){
            reducedB[i*(K/2) + j] = matB[i*K + (j*2)] + matB[i*K + (j*2 + 1)];
        }
    }


    void *ptrC = nullptr;
    cudaError_t gpu_matC = cudaMalloc(&ptrC, (M/2) * (K/2) * sizeof(int));
    if(gpu_matC != cudaSuccess) {
        cout << "Error allocating memory for matC on device: " << cudaGetErrorString(gpu_matC) << endl;
        return;
    }

    void *ptrA_reduced = nullptr;
    cudaError_t gpu_matA_reduced = cudaMalloc(&ptrA_reduced, (M/2) * N * sizeof(int));
    if(gpu_matA_reduced != cudaSuccess) {
        cout << "Error allocating memory for matA_reduced on device: " << cudaGetErrorString(gpu_matA_reduced) << endl;
        return;
    }

     void *ptrB_reduced = nullptr;
    cudaError_t gpu_matB_reduced = cudaMalloc(&ptrB_reduced, N * (K/2) * sizeof(int));
    if(gpu_matB_reduced != cudaSuccess) {
        cout << "Error allocating memory for matB_reduced on device: " << cudaGetErrorString(gpu_matB_reduced) << endl;
        return;
    }


    cudaEventRecord(cpy_H2D_start);
    /* Copying array(s) from host to device goes here */

    cudaError_t copyA = cudaMemcpy(ptrA_reduced, reducedA, (M/2) * N * sizeof(int), cudaMemcpyHostToDevice);
    if(copyA != cudaSuccess) {
        cout << "Error copying matA_reduced from host to device: " << cudaGetErrorString(copyA) << endl;
        return;
    }

    cudaError_t copyB = cudaMemcpy(ptrB_reduced, reducedB, N * (K/2) * sizeof(int), cudaMemcpyHostToDevice);
    if(copyB != cudaSuccess) {
        cout << "Error copying matB_reduced from host to device: " << cudaGetErrorString(copyB) << endl;
        return;
    }

    
    


    cudaEventRecord(cpy_H2D_end);
    cudaEventSynchronize(cpy_H2D_end);

    cudaEventRecord(comp_start);
    /* Launching the GPU kernel to do the computation goes here */

    dim3 blockDim(16, 16);
    dim3 gridDim((K/2 + blockDim.x - 1) / blockDim.x, 
                 (M/2 + blockDim.y - 1) / blockDim.y);

    rmm_kernel<<<gridDim, blockDim>>>(static_cast<int*>(ptrA_reduced), static_cast<int*>(ptrB_reduced), static_cast<int*>(ptrC), M, N, K);

    cudaEventRecord(comp_end);
    cudaEventSynchronize(comp_end);

    cudaEventRecord(cpy_D2H_start);
    /* Copying array(s) from device to host goes here */
    cudaError_t copyC = cudaMemcpy(matC, ptrC, (M/2) * (K/2) * sizeof(int), cudaMemcpyDeviceToHost);
    if(copyC != cudaSuccess) { 
        cout << "Error copying matC from device to host: " << cudaGetErrorString(copyC) << endl;
        return;
    }
    cudaEventRecord(cpy_D2H_end);
    cudaEventSynchronize(cpy_D2H_end);

    /* Postprocessing (if any) goes here */
    cudaError_t freeA = cudaFree(ptrA_reduced);
    if(freeA != cudaSuccess) {
        cout << "Error freeing memory for matA_reduced on device: " << cudaGetErrorString(freeA) << endl;
    }

    cudaError_t freeB = cudaFree(ptrB_reduced);
    if(freeB != cudaSuccess) {
        cout << "Error freeing memory for matB_reduced on device: " << cudaGetErrorString(freeB) << endl;
    }

    cudaError_t freeC = cudaFree(ptrC);
    if(freeC != cudaSuccess) {
        cout << "Error freeing memory for matC on device: " << cudaGetErrorString(freeC) << endl;
    }

    /* Display timing statistics */
    float time;
    cudaEventElapsedTime(&time, cpy_H2D_start, cpy_H2D_end);
    cout << "Host to Device MemCpy takes " << setprecision(4) << time/1000 << "s" << endl;

    cudaEventElapsedTime(&time, comp_start, comp_end);
    cout << "RMM operation takes " << setprecision(4) << time/1000 << "s" << endl;

    cudaEventElapsedTime(&time, cpy_D2H_start, cpy_D2H_end);
    cout << "Device to Host MemCpy takes " << setprecision(4) << time/1000 << "s" << endl;
}