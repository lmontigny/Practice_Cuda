__global__ void Kernel(const int count)
{
    extern __shared__ int a[];
}


Kernel<<< gridDim, blockDim, a_size >>>(count)
