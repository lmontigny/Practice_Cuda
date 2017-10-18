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

![alt text](highres_255807524.jpeg)
