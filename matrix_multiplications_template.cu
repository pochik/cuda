#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cublas_v2.h>

#define BLOCK_SIZE 32

__global__ void gpu_matrix_mult_global(float *a,
                                       float *b,
                                       float *c,
                                       int m,
                                       int n,
                                       int k)
{
    // TODO
}

__global__ void gpu_square_matrix_mult(float *d_a, float *d_b, float *d_result, int n)
{
    // TODO
}

void cpu_matrix_mult(float *h_a, float *h_b, float *h_result, int m, int n, int k)
{
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            float tmp = 0.0;
            for (int h = 0; h < n; ++h)
            {
                tmp += h_a[i * n + h] * h_b[h * k + j];
            }
            h_result[i * k + j] = tmp;
        }
    }
}

int main(int argc, char const *argv[])
{
    int m, n, k;
    /* Fixed seed for illustration */
    srand(3333);
    m = atoi(argv[1]);
    n = atoi(argv[1]);
    k = atoi(argv[1]);
    
    // allocate memory in host RAM, h_cc is used to store CPU result
    float *h_a, *h_b, *h_c, *h_cc;
    cudaMallocHost((void **) &h_a, sizeof(float)*m*n);
    cudaMallocHost((void **) &h_b, sizeof(float)*n*k);
    cudaMallocHost((void **) &h_c, sizeof(float)*m*k);
    cudaMallocHost((void **) &h_cc, sizeof(float)*m*k);
    
    // random initialize matrix A
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            h_a[i * n + j] = rand() % 1024;
        }
    }
    
    // random initialize matrix B
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            h_b[i * k + j] = rand() % 1024;
        }
    }
    
    float gpu_elapsed_time_ms, cpu_elapsed_time_ms;
    
    // some events to count the execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // start to count execution time of GPU version
    // Allocate memory space on the device
    float *d_a, *d_b, *d_c;
    cudaMalloc((void **) &d_a, sizeof(float)*m*n);
    cudaMalloc((void **) &d_b, sizeof(float)*n*k);
    cudaMalloc((void **) &d_c, sizeof(float)*m*k);
    
    // copy matrix A and B from host to device memory
    cudaMemcpy(d_a, h_a, sizeof(float)*m*n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(float)*n*k, cudaMemcpyHostToDevice);
    
    // TODO compute grid size
    
    cudaEventRecord(start, 0);
    
    if(atoi(argv[2]) == 1)
    {
        // TODO call shared memory version
    }
    else if(atoi(argv[2]) == 2)
    {
        // TODO call global memory version
    }
    
    // Transefr results from device to host
    cudaDeviceSynchronize();
    // time counting terminate
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    cudaMemcpy(h_c, d_c, sizeof(float)*m*k, cudaMemcpyDeviceToHost);
    
    // compute time elapse on GPU computing
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    printf("Time elapsed on matrix multiplication of %dx%d . %dx%d on GPU: %f ms.\n\n", m, n, n, k, gpu_elapsed_time_ms);
    
    // start the CPU version
    cudaEventRecord(start, 0);
    
    cpu_matrix_mult(h_a, h_b, h_cc, m, n, k);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cpu_elapsed_time_ms, start, stop);
    printf("Time elapsed on matrix multiplication of %dx%d . %dx%d on CPU: %f ms.\n\n", m, n, n, k, cpu_elapsed_time_ms);
    
    // validate results computed by GPU
    int all_ok = 1;
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            if(h_cc[i*k + j] != h_c[i*k + j])
            {
                all_ok = 0;
            }
        }
    }
    
    // roughly compute speedup
    if(all_ok)
    {
        printf("all results are correct!!!, speedup = %f\n", cpu_elapsed_time_ms / gpu_elapsed_time_ms);
    }
    else
    {
        printf("incorrect results\n");
    }
    
    // free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    cudaFreeHost(h_cc);
    
    return 0;
}
