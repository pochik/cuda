#include <iostream>
#include <cuda_runtime.h>
 
using namespace std;
 
__global__ void transp(int *A, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n*n){
        int j = idx % n;
        int i = idx / n;
 
        int tmp = A[i * n + j];
        A[i * n + j] = A[j * n + i];
        A[j * n + i] = tmp;
    }
}
 
bool check(int *c, int *t, int n){
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            cout << c[i * n + j] << " " << t[j * n + i] << endl;
            if (c[i * n + j] != t[j * n + i]) return false;
        }
    }
    return true;
}
 
int main(int argc, char **argv){
 
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
 
    int n = atoi(argv[1]);
    int *h_a = (int *) malloc(n * n * sizeof(int));
    int *h_b = (int *) malloc(n * n * sizeof(int));
 
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            h_a[i * n + j] = i * n + j + 1;
            h_b[i * n + j] = i * n + j + 1;
            //cout << h_a[i * n + j] << " ";
        }
        //cout << endl;
    }
    cout << endl;
 
    int *d_a;
 
    cudaEventRecord(start);
 
    cudaMalloc(&d_a, n * n * sizeof(int));
 
    cudaMemcpy(d_a, h_a, n * n * sizeof(int), cudaMemcpyHostToDevice);
 
    int blockSize, gridSize;
    blockSize = 1024;
    gridSize = (n - 1) / 1024 + 1;
 
    transp<<<gridSize, blockSize>>>(d_a, n);
 
    cudaDeviceSynchronize();
 
    cudaMemcpy(h_a, d_a, n * n * sizeof(int), cudaMemcpyDeviceToHost);
 
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
 
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
 
    cout << milliseconds << endl;
 
    cudaFree(d_a);
 
    if (check(h_a, h_b, n) == true) cout << "OK\n";
    else cout << "Incorrect";
 
    /*
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            cout << h_a[i * n + j] << " ";
        }
        cout << endl;
    }
    */
 
    return 0;
}
