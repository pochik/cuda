//
//  main.cpp
//  
//
//  Created by Elijah Afanasiev on 25.09.2018.
//
//

// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <chrono>
#include <cuda_profiler_api.h>
#include <iostream>

using namespace std;

#ifndef MAX
#define MAX(a,b) (a > b ? a : b)
#endif

__global__ void vectorAddGPU(float *a, float *b, float *c, int N)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (idx < N)
    {
        c[idx] = a[idx] + b[idx];
    }
}

void unified_sample(int size = 1048576)
{   
    printf("UNIFIED SAMPLE:\n");
    int n = size;
    
    int nBytes = n*sizeof(float);
    
    float *a, *b;  // host data
    float *c;  // results
    
    dim3 block(256);
    dim3 grid((unsigned int)ceil(n/(float)block.x));
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    cudaMallocManaged(&a, nBytes);
    cudaMallocManaged(&b, nBytes);
    cudaMallocManaged(&c, nBytes);

    for(int i=0;i<n;i++)
    {
        a[i] = rand() / (float)RAND_MAX;
        b[i] = rand() / (float)RAND_MAX;
        c[i] = 0;
    }
    
    vectorAddGPU<<<grid, block>>>(a, b, c, n);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("time: %f ms\n", milliseconds);
    
    cudaThreadSynchronize();
    
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
}

void pinned_sample(int size = 1048576)
{   
    printf("PINNED SAMPLE:\n");
    int n = size;
    
    int nBytes = n*sizeof(float);
    
    float *a, *b;  // host data
    float *c;  // results
    
    auto start1 = std::chrono::steady_clock::now();
    a = (float *)malloc(nBytes);
    b = (float *)malloc(nBytes);
    c = (float *)malloc(nBytes);
    cudaHostRegister(a, nBytes, 0);
    cudaHostRegister(b, nBytes, 0);
    cudaHostRegister(c, nBytes, 0);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start1;
    cout << "Pinned host malloc time: " << (elapsed_seconds.count())*1000.0 << " ms" << endl;
    
    float *a_d,*b_d,*c_d;
    
    dim3 block(256);
    dim3 grid((unsigned int)ceil(n/(float)block.x));
    
    for(int i=0;i<n;i++)
    {
        a[i] = rand() / (float)RAND_MAX;
        b[i] = rand() / (float)RAND_MAX;
        c[i] = 0;
    }
    
    start1 = std::chrono::steady_clock::now();
    cudaMalloc((void **)&a_d,n*sizeof(float));
    cudaMalloc((void **)&b_d,n*sizeof(float));
    cudaMalloc((void **)&c_d,n*sizeof(float));
    end = std::chrono::steady_clock::now();
    elapsed_seconds = end-start1;
    cout << "Pinned device malloc time: " << (elapsed_seconds.count())*1000.0 << " ms" << endl;

    
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);

    start1 = std::chrono::steady_clock::now();
    cudaMemcpy(a_d,a,n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d,b,n*sizeof(float), cudaMemcpyHostToDevice);
    end = std::chrono::steady_clock::now();
    elapsed_seconds = end-start1;
    cout << "Pinned copy time: " << (elapsed_seconds.count())*1000.0 << " ms" << endl;
    
    vectorAddGPU<<<grid, block>>>(a_d, b_d, c_d, n);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("time: %f ms\n", milliseconds);
    
    cudaThreadSynchronize();
    
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    cudaHostUnregister(a);
    cudaHostUnregister(b);
    cudaHostUnregister(c);
    free(a);
    free(b);
    free(c);
}

void usual_sample(int size = 1048576)
{   
    printf("USUAL SAMPLE:\n");
    int n = size;
    
    int nBytes = n*sizeof(float);
    
    float *a, *b;  // host data
    float *c;  // results
    
    auto start1 = std::chrono::steady_clock::now();
    a = (float *)malloc(nBytes);
    b = (float *)malloc(nBytes);
    c = (float *)malloc(nBytes);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start1;
    cout << "Usual host malloc time: " << (elapsed_seconds.count())*1000.0 << " ms" << endl;
    
    float *a_d,*b_d,*c_d;
    
    dim3 block(256);
    dim3 grid((unsigned int)ceil(n/(float)block.x));
    
    for(int i=0;i<n;i++)
    {
        a[i] = rand() / (float)RAND_MAX;
        b[i] = rand() / (float)RAND_MAX;
        c[i] = 0;
    }
    
    
    start1 = std::chrono::steady_clock::now();
    cudaMalloc((void **)&a_d,n*sizeof(float));
    cudaMalloc((void **)&b_d,n*sizeof(float));
    cudaMalloc((void **)&c_d,n*sizeof(float));
    end = std::chrono::steady_clock::now();
    elapsed_seconds = end-start1;
    cout << "Usual device malloc time: " << (elapsed_seconds.count())*1000.0 << " ms" << endl;
    
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);

    start1 = std::chrono::steady_clock::now();
    cudaMemcpy(a_d,a,n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d,b,n*sizeof(float), cudaMemcpyHostToDevice);
    end = std::chrono::steady_clock::now();
    elapsed_seconds = end-start1;
    cout << "Usual copy time: " << (elapsed_seconds.count())*1000.0 << " ms" << endl;
    
    
    vectorAddGPU<<<grid, block>>>(a_d, b_d, c_d, n);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("time: %f ms\n", milliseconds);
    
    cudaThreadSynchronize();
    
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
}


int main(int argc, char **argv)
{
    int n = atoi(argv[1]);
    usual_sample(n);
    printf("\n");
    pinned_sample(n);
    printf("\n");
    unified_sample(n);
    
    return 0;
}
