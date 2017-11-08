/*
However, if you are combining MPI and CUDA, you often need to send GPU
buffers instead of host buffers. Without CUDA-aware MPI, you need to 
stage GPU buffers through host memory, using cudaMemcpy as shown in 
the following code excerpt.
*/

//MPI rank 0
cudaMemcpy(s_buf_h,s_buf_d,size,cudaMemcpyDeviceToHost);
MPI_Send(s_buf_h,size,MPI_CHAR,1,100,MPI_COMM_WORLD);

//MPI rank 1
MPI_Recv(r_buf_h,size,MPI_CHAR,0,100,MPI_COMM_WORLD, &status);
cudaMemcpy(r_buf_d,r_buf_h,size,cudaMemcpyHostToDevice);

// With a CUDA-aware MPI library this is not necessary; the GPU buffers can be directly passed to MPI as in the following excerpt.

//MPI rank 0
MPI_Send(s_buf_d,size,MPI_CHAR,1,100,MPI_COMM_WORLD);

//MPI rank n-1
MPI_Recv(r_buf_d,size,MPI_CHAR,0,100,MPI_COMM_WORLD, &status);
