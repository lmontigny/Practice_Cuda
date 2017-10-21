__global__ void update (float *u, float *u_prev, int N, float dx, float dt, float c)
{
  // Each thread will load one element
  int i = threadIdx.x;
  int I = threadIdx.x + BLOCKSIZE * blockIdx.x;
  __shared__ float u_shared[BLOCKSIZE];
  
  if (I>=N){return;}
  u_shared[i] = u[I];
  __syncthreads();
  
  if (i>0 && i<BLOCKSIZE‐1)
  { 
    u[I] = u_shared[i] ‐ c*dt/dx*(u_shared[i] ‐ u_shared[i‐1]);
  }
  else
  { 
    u[I] = u_prev[I] ‐ c*dt/dx*(u_prev[I] ‐ u_prev[I‐1]);
  }
}
