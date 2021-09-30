// UCSC CMPE220 Advanced Parallel Processing 
// Prof. Heiner Leitz
// Author: Marcelo Siero.
// Modified from code by:: Andreas Goetz (agoetz@sdsc.edu)
// CUDA program to perform 1D stencil operation in parallel on the GPU
//
// /* FIXME */ COMMENTS ThAT REQUIRE ATTENTION

#include <iostream>
#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <sys/time.h>

// define vector length, stencil radius, 
#define N (1024*1024*512l)
// #define RADIUS 3
#define GRIDSIZE 128
#define BLOCKSIZE 256

int gridSize  = GRIDSIZE;
int blockSize = BLOCKSIZE;

void cudaErrorCheck() {
    // FIXME: Add code that finds the last error for CUDA functions performed.
    // Upon getting an error have it print out a meaningful error message as 
    //  provided by the CUDA API, then exit with an error exit code.
}


cudaEvent_t start, stop;
struct timeval cpu_start, cpu_stop;
void start_timer(int mode) {
    // FIXME: ADD TIMING CODE, HERE, USE GLOBAL VARIABLES AS NEEDED.
    if (mode) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
    } else {
        gettimeofday(&cpu_start, NULL);
    }
}

float stop_timer(int mode) {
    // FIXME: ADD TIMING CODE, HERE, USE GLOBAL VARIABLES AS NEEDED.
    if (mode) {
        cudaEventRecord(stop);
        float gpu_time = 0;
        cudaEventElapsedTime(&gpu_time, start, stop);
        return gpu_time;
    } else {
        gettimeofday(&cpu_stop, NULL);
        double cpu_time = (cpu_stop.tv_sec - cpu_start.tv_sec)*1000.0 + (cpu_stop.tv_usec - cpu_start.tv_usec)/1000.0;
        return cpu_time;
    }
}

cudaDeviceProp prop;
void getDeviceProperties() {
    // FIXME: Implement this function so as to acquire and print the following 
    // device properties:
    //    Major and minor CUDA capability, total device global memory,
    //    size of shared memory per block, number of registers per block,
    //    warp size, max number of threads per block, number of multi-prccessors
    //    (SMs) per device, Maximum number of threads per block dimension (x,y,z),
    //    Maximumum number of blocks per grid dimension (x,y,z).
    //
    // These properties can be useful to dynamically optimize programs.  For
    // instance the number of SMs can be useful as a heuristic to determine
    // how many is a good number of blocks to use.  The total device global
    // memory might be important to know just how much data to operate on at
    // once.
}

void newline() { std::cout << std::endl; };

void printThreadSizes() {
    int noOfThreads = gridSize * blockSize;
    printf("Blocks            = %d\n", gridSize);  // no. of blocks to launch.
    printf("Threads per block = %d\n", blockSize); // no. of threads to launch.
    printf("Total threads     = %d\n", noOfThreads);
    printf("Number of grids   = %d\n", (N + noOfThreads -1)/ noOfThreads);
}

// -------------------------------------------------------
// CUDA device function that performs 1D stencil operation
// -------------------------------------------------------
__global__ void stencil_1D(double *in, double *out, long dim, int radius){
    int elements_per_block = dim/gridDim.x + 1;
    int block_offset = elements_per_block * blockIdx.x;

    extern __shared__ double sh_mem[];
    int el_per_thr_load = (elements_per_block + 2*radius)/blockDim.x + 1;
    int cur_sh_id = el_per_thr_load * threadIdx.x;
    int next_sh_id = el_per_thr_load * (threadIdx.x + 1);
    for(int i = cur_sh_id; i < next_sh_id && i < elements_per_block + 2*radius; i++) {
        if(block_offset - radius + i < dim && block_offset - radius + i > -1)
            sh_mem[i] = in[block_offset - radius + i];
        else
            sh_mem[i] = 0;
    }
    __syncthreads();

    int elements_per_thread = elements_per_block/blockDim.x + 1;
    double res = 0;
    for (int j = 0; j < 2*radius; j++) {
        res += sh_mem[j + threadIdx.x * elements_per_thread];
    }
    int ind = block_offset + threadIdx.x * elements_per_thread;
    if (ind < dim)
            out[ind] = res;
    ind++;
    for (int i = 1; i < elements_per_thread && ind < dim; i++, ind++) {
        int before_radius_id = (i - 1) + threadIdx.x * elements_per_thread;
        res -= sh_mem[before_radius_id];
        res += sh_mem[before_radius_id + 2*radius + 2];
        out[ind] = res;
    }

}


__global__ void stencil_1D_old(double *in, double *out, long dim, int radius){

    long gindex = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = gridDim.x * blockDim.x;
  
    // Go through all data
    // Step all threads in a block to avoid synchronization problem
    while ( gindex < (dim + blockDim.x) ) {
  
        /* FIXME PART 2 - MODIFIY PROGRAM TO USE SHARED MEMORY. */
    
        // Apply the stencil
        double result = 0;
        for (int offset = -radius; offset <= radius; offset++) {
            if ( gindex + offset < dim && gindex + offset > -1)
                result += in[gindex + offset];
        }
  
    // Store the result
    if (gindex < dim)
        out[gindex] = result;
  
    // Update global index and quit if we are done
    gindex += stride;
  
    __syncthreads();
  
    }
}


#define True  1
#define False 0
void checkResults(double *h_in, double *h_out, int radius, int DoCheck=True) {
    // DO NOT CHANGE THIS CODE.
    // CPU calculates the stencil from data in *h_in
    // if DoCheck is True (default) it compares it with *h_out
    // to check the operation of this code.
    // If DoCheck is set to False, it can be used to time the CPU.
    int i, j, ij;
    double result, err;
    err = 0;
    for (i=0; i<N; i++){  // major index.
        result = 0;
        for (j=-radius; j<=radius; j++){
            ij = i+j;
            if (ij>=0 && ij<N)
                result += h_in[ij];
        }
        if (DoCheck) {  // print out some errors for debugging purposes.
            if (h_out[i] - result > 0.000001) { // count errors.
                err++;
                if (err < 8) { // help debug
                printf("h_out[%d]=%lf should be %lf\n",i,h_out[i], result);
                };
            }
        } else {  // for timing purposes.
            h_out[i] = result;
        }
    }

    if (DoCheck) { // report results.
        if (err != 0){
            printf("Error, %lf elements do not match!\n", err);
        } else {
            printf("Success! All elements match CPU result.\n");
        }
    }
}

// ------------
// main program
// ------------
int main(int argc, char **argv)
{
    int radius = argc < 2 ? 3 : atoi(argv[1]);
    double *h_in, *h_out;
    double *d_in, *d_out;
    long size = N * sizeof(double);
    int i;

    // allocate host memory
    h_in = (double *)malloc(size);
    h_out = (double *)malloc(size);

    getDeviceProperties();

    // initialize vector
    for (i=0; i<N; i++){
        //    h_in[i] = i+1;
        h_in[i] = 1;
    }

    // allocate device memory
    cudaMalloc((void **)&d_in, size);
    cudaMalloc((void **)&d_out, size);
    cudaErrorCheck();

    // copy input data to device
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
    cudaErrorCheck();


    // Apply stencil by launching a sufficient number of blocks
    printf("\n---------------------------\n");
    printf("Launching old 1D stencil kernel\n");
    printf("---------------------------\n");
    printf("Vector length     = %ld (%ld MB)\n",N,N*sizeof(double)/1024/1024);
    printf("Stencil radius    = %d\n",radius);

    //----------------------------------------------------------
    // CODE TO RUN AND TIME THE STENCIL KERNEL.
    //----------------------------------------------------------
    newline();
    printThreadSizes();

    
    start_timer(1);
    stencil_1D_old<<<gridSize, blockSize>>>(d_in, d_out, N, radius);
    cudaDeviceSynchronize();
    std::cout << "Elapsed time of old kernel: " << stop_timer(1) << std::endl;

    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);

    // copy results back to host
    cudaErrorCheck();
    checkResults(h_in, h_out, radius);


    // Apply stencil by launching a sufficient number of blocks
    printf("\n---------------------------\n");
    printf("Launching new 1D stencil kernel\n");
    printf("---------------------------\n");
    printf("Vector length     = %ld (%ld MB)\n",N,N*sizeof(double)/1024/1024);
    printf("Stencil radius    = %d\n",radius);

    //----------------------------------------------------------
    // CODE TO RUN AND TIME THE STENCIL KERNEL.
    //----------------------------------------------------------
    newline();
    printThreadSizes();

    
    start_timer(1);
    stencil_1D<<<gridSize, blockSize, (N/gridSize + 1 + 2*radius) * sizeof(double)>>>(d_in, d_out, N, radius);
    cudaDeviceSynchronize();
    std::cout << "Elapsed time of new kernel: " << stop_timer(1) << std::endl;

    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);

    // copy results back to host
    cudaErrorCheck();
    checkResults(h_in, h_out, radius);
    //----------------------------------------------------------

    // deallocate device memory
    cudaFree(d_in);
    cudaFree(d_out);
    cudaErrorCheck();
    //=====================================================
    // Evaluate total time of execution with just the CPU.
    //=====================================================
    newline();
    std::cout << "Running stencil with the CPU.\n";
    start_timer(0);
    // Use checkResults to time CPU version of the stencil with False flag.
    checkResults(h_in, h_out, radius, False);
    std::cout << "Elapsed time: " << stop_timer(0) << std::endl;
    //=====================================================

    // deallocate host memory
    free(h_in);
    free(h_out);

    return 0;
}
