#include <stdio.h>  
#include <cuda_runtime.h>  
#include "helper_cuda.h"  

/* A very simple kernel function */
 __global__ void kernel(int *d_var) { d_var[threadIdx.x] += 10; } 
 
 int * host_p;  
 int * host_result;  
 int * dev_p;  
 
int main(void) {  
      int ns = 4;  
      int data_size = ns * sizeof(int);
      
      /* Allocate host_p as pinned memory */
      checkCudaErrors( 
        cudaHostAlloc((void**)&host_p, data_size, 
        cudaHostAllocDefault) );  
      
      /* Allocate host_result as pinned memory */
      checkCudaErrors( 
        cudaHostAlloc((void**)&host_result, data_size, 
        cudaHostAllocDefault) );  
      /* Allocate dev_p on the device global memory */
      checkCudaErrors( 
        cudaMalloc((void**)&dev_p, data_size) );  
      
      /* Initialise host_p*/
      for (int i=0; i<ns; i++){  
           host_p[i] = i + 1;  
      }  
      
      /* Transfer data to the device host_p .. dev_p */
      checkCudaErrors( 
        cudaMemcpy(dev_p, host_p, data_size, cudaMemcpyHostToDevice) );
      
      /* Now launch the kernel... */
      kernel<<<1,  ns>>>(dev_p);  
      getLastCudaError("Kernel error");
      
      /* Copy the result from the device back to the host */
      checkCudaErrors( 
        cudaMemcpy(host_result, dev_p, data_size, cudaMemcpyDeviceToHost) );
      
      /* and print the result */
      for (int i=0; i<ns; i++){  
           printf("result[%d] = %d\n", i, host_result[i]);  
      }  
      
      /*
       * Now free the memory!
       */
      checkCudaErrors( cudaFree(dev_p) );  
      checkCudaErrors( cudaFreeHost(host_p) );  
      checkCudaErrors( cudaFreeHost(host_result) );  
      
      return 0;  
 } 
