#include <iostream>
#include <cuda_runtime.h>
 
using namespace std;
 
__global__ void sum_kernel(double* A, double* B, double* C, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        double a = A[idx];
        double b = B[idx];
        C[idx] = a + b;
    }
}
 
int main(int argc, char **argv){
 
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
 
 
    int n = atoi(argv[1]);
 
    double *h_a, *h_b, *h_c;
 
    size_t bytes = n * sizeof(double);
 
    h_a = (double *) malloc(bytes);
    h_b = (double *) malloc(bytes);
    h_c = (double *) malloc(bytes);
 
    for (int i = 0; i < n; i++){
        h_a[i] = sin(i) * sin(i);
        h_b[i] = cos(i) * cos(i);
    }
 
    double *d_a, *d_b, *d_c;
 
    cudaEventRecord(start);
 
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
 
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
 
    int blockSize, gridSize;
    blockSize = 1024;
    gridSize = (n - 1) / 1024 + 1;
 
    sum_kernel<<<gridSize, blockSize>>> (d_a, d_b, d_c, n);
 
    cudaDeviceSynchronize();
 
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
 
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
 
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
 
    cout << milliseconds << endl;
 
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
 
/*
    for (int i = 0; i < n; i++){
        cout << h_c[i] << endl;
    }
*/
    return 0;
}
