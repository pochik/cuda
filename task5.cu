#include <iostream>
#include <cuda_runtime.h>
 
 
using namespace std;
 
 
__global__ void reverse_kernel(int *a, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size / 2){
        return;
    }
    extern __shared__ int a_copy[];
    a_copy[idx] = a[idx];
    a_copy[size - 1 - idx] = a[size - 1 - idx];
    __syncthreads();
    int t = a_copy[idx];
    a_copy[idx] = a_copy[size - 1 - idx];
    a_copy[size - 1 - idx] = t;
    __syncthreads();
    a[idx] = a_copy[idx];
    a[size - 1 - idx] = a_copy[size - 1 - idx];
}
 
int main(int argc, char **argv){
    int n = atoi(argv[1]);
 
    int *h_a;
    size_t bytes = n  * sizeof(int);
 
    h_a = (int *) malloc(bytes);
 
    for (int i = 0; i  < n; i++){
        h_a[i] = i;
    }
 
    int *d_a;
 
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
 
    cudaEventRecord(start);
 
    cudaMalloc(&d_a, bytes);
 
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
 
    int blockSize = 1024;
    int gridSize = 1;
 
    reverse_kernel<<<gridSize, blockSize>>>(d_a, n);
    cudaDeviceSynchronize();
 
    cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost);
 
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
 
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
 
    cout << milliseconds << endl;
 
    cudaFree(d_a);
 
    return 0;
}
