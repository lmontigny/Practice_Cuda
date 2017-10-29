int main(void) {

 int * host_p; /*< Host data allocated as pinned memory */
 int * dev_ptr_p; /*< this pointer resides on the host */
 int ns = 32;
 int data_size = ns * sizeof(int);

 checkCudaErrors(
   cudaHostAlloc((void**) &host_p, data_size, cudaHostAllocMapped));

 /* host_p = {1, 2, 3, ..., ns}*/
 for (int i = 0; i < ns; i++)
  host_p[i] = i + 1;

 /*
  * we can pass the address of `host_p`,
  * namely `dev_ptr_p` to the kernel. This address
  * is retrieved using cudaHostGetDevicePointer:
  * */
 checkCudaErrors(cudaHostGetDevicePointer(&dev_ptr_p, host_p, 0));
 kernel<<<1, ns>>>(dev_ptr_p);

 /*
  * The following line is necessary for the host
  * to be able to "see" the changes that have been done
  * on `host_p`
  */
 checkCudaErrors(cudaDeviceSynchronize());

 for (int i = 0; i < ns; i++)
  printf("host_p[%d] = %d\n", i, host_p[i]);

 /* Free the page-locked memory */
 checkCudaErrors(cudaFreeHost(host_p));
 
 return 0;
}
