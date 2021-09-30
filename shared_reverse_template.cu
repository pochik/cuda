#include <stdio.h>
#include <stdlib.h>

__global__ void staticReverse(int *d, int n)
{
  /* FIX ME */
}

__global__ void dynamicReverse(int *d, int n)
{
  /* FIX ME */
}

int main(void)
{
  const int n = 64; // FIX ME TO max possible size
  int a[n], r[n], d[n]; // FIX ME TO dynamic arrays if neccesary

  for (int i = 0; i < n; i++) {
    a[i] = i;
    r[i] = n-i-1;
    d[i] = 0;
  }

  int *d_d;
  cudaMalloc(&d_d, n * sizeof(int)); 

  // run version with static shared memory
  cudaMemcpy(d_d, a, n*sizeof(int), cudaMemcpyHostToDevice);
  staticReverse<<<...>>>(d_d, n); // FIX kernel execution params
  cudaMemcpy(d, d_d, n*sizeof(int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; i++) 
    if (d[i] != r[i]) printf("Error: d[%d]!=r[%d] (%d, %d)n", i, i, d[i], r[i]);

  // run dynamic shared memory version
  cudaMemcpy(d_d, a, n*sizeof(int), cudaMemcpyHostToDevice);
  dynamicReverse<<<...>>>(d_d, n); // FIX kernel executon params
  cudaMemcpy(d, d_d, n * sizeof(int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; i++) 
    if (d[i] != r[i]) printf("Error: d[%d]!=r[%d] (%d, %d)n", i, i, d[i], r[i]);
}
