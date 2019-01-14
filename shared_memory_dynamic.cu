__global__ void Kernel(int count_a, int count_b)
{
    extern __shared__ int *shared;
    int *a = &shared[0]; //a is manually set at the beginning of shared
    int *b = &shared[count_a]; //b is manually set at the end of a
}

sharedMemory = count_a*size(int) + size_b*size(int);
Kernel <<<numBlocks, threadsPerBlock, sharedMemory>>> (count_a, count_b);
