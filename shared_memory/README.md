# Shared Memory

## GPU Information
GTX 1070

Maximum number of threads per block: 1024

Max dimension size of a thread block (x,y,z): (1024, 1024, 64)

Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535

## Launch kernel
```javascript
<<< #blocks, #threads per block >>>
int blockSize = 256;
int numBlocks = (N + blockSize - 1) / blockSize;
add<<<numBlocks, blockSize>>>(N, x, y);
```

## Shared Memory
Portion of L1 cache
Shard by threads within a block
Consider race condition (ex: two threads changing the same variable to different value)

```javascript
// Static
__global__ void myKernel{
__shared__ int i;
__shared__ float f_array[100]; // size specified at compile time
}

// Dynamic
int M = 10; // size of shared memory
myKernel<<<10,32,M>>>
__global__ void myKernel{
extern __shared__ char shared_buffer[];   // length going to be M 
}
```

## Race condtion
!! Very bad !!
```javascript
// Static
__shared__ int i;
i = threadIdx.x;
}
```

Use syncthreads(), it is a block-wide barrier.
Solution to the previous issue:

```javascript
// Static
__shared__ int i;
i = 0;
__syncthreads()
if(threadIdx.x==0) i++;
__syncthreads()
if(threadIdx.x==1) i++;
}
```

## Wraps
Threads of a block are executed in groups called wraps
1 wraps = 32 threads

## Blocking
