__global__ void Kernel(int count_a, int count_b)
{
    __shared__ int a[100];
    __shared__ int b[4];
}
