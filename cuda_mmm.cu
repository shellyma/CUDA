#include <cstdlib>
#include <cstdio>
#include <math.h>

// Assertion to check for errors
#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
  if (code != cudaSuccess){
    fprintf(stderr,"CUDA_SAFE_CALL: %s %s %d\n", cudaGetErrorString(code), file, line);
  if (abort) exit(code);
   }
}
#define LOOP    2000
#define TOL     1e-6
#define ARR_LEN 2000
#define NUM_THREAD 16
#define TILE_WIDTH 8

__global__ void kernel_mmm_global(float *d_A, float *d_B, float *d_C) {
    int index_y = TILE_WIDTH*blockIdx.x + threadIdx.x;
    int index_x = TILE_WIDTH*blockIdx.y + threadIdx.y;
    int iter;
    
    for(iter =0; iter < ARR_LEN; iter++){
         d_C[index_x*ARR_LEN + index_y] += d_A[index_x*ARR_LEN + iter] * d_B[iter*ARR_LEN + index_y]; 
  }
}

__global__ void kernel_mmm_shared(float *d_A, float *d_B, float *d_C) {

    __shared__ float A_tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ float B_tile[TILE_WIDTH][TILE_WIDTH];

    int index_y = TILE_WIDTH * blockIdx.x  + threadIdx.x;
    int index_x = TILE_WIDTH * blockIdx.y  + threadIdx.y;
  	
    int index = index_x * ARR_LEN + index_y;
    int m, k;
#pragma unroll
    for(m=0; m<ARR_LEN/TILE_WIDTH;m++){

	A_tile[threadIdx.y][threadIdx.x] = d_A[index_x *ARR_LEN + (m*TILE_WIDTH + threadIdx.x)];
	B_tile[threadIdx.y][threadIdx.x] = d_B[(m*TILE_WIDTH + threadIdx.y) + ARR_LEN + index_y];
	__syncthreads();

	for( k=0; k<TILE_WIDTH;k++)
		d_C[index] += A_tile[threadIdx.x][k] * B_tile[k][threadIdx.y];
	__syncthreads();
    }


}


main (int argc, char **argv) {

    float A[ARR_LEN][ARR_LEN], B[ARR_LEN][ARR_LEN], C[ARR_LEN][ARR_LEN];
    float *d_A, *d_B, *d_C,*d_C1; // These are the copies of A and B on the GPU
    float *h_C, *h_C1; // This is a host copy of the output of B from the GPU to CPU
    int i, j,k,iter;
    int num_bytes = ARR_LEN * ARR_LEN * sizeof(float);
    int errCount = 0;
    float accu1;
    // Input is randomly generated
  	for(i=0;i<ARR_LEN;i++) {
             for(j=0;j<ARR_LEN;j++) {
           	 A[i][j] = (float) (rand() / 2 +.5f);
                 B[i][j] = (float) (rand() / 3 +.5f);
		}
         }
   

    cudaEvent_t start_event0, stop_event0;
    float elapsed_time_cpu;
    CUDA_SAFE_CALL( cudaEventCreate(&start_event0) );
    CUDA_SAFE_CALL( cudaEventCreate(&stop_event0) );
    cudaEventRecord(start_event0, 0);
    // CPU compuation
    for (k = 0; k < ARR_LEN; k++) {
      for (i = 0; i < ARR_LEN; i++) {
     accu1 = A[i][k];
       for (j = 0; j < ARR_LEN; j++)
        C[i][j] += accu1* B[k][j];
       }
    }
  
    cudaEventRecord(stop_event0, 0);
    cudaEventSynchronize(stop_event0);
    CUDA_SAFE_CALL( cudaEventElapsedTime(&elapsed_time_cpu,start_event0, stop_event0) );



    h_C = (float *)malloc(num_bytes);
    memset(h_C, 0, num_bytes);
    h_C1 = (float *)malloc(num_bytes);
    memset(h_C1, 0, num_bytes);
    //ALLOCATE MEMORY FOR GPU COPIES OF A AND B
    cudaMalloc((void**)&d_A, num_bytes);
    cudaMalloc((void**)&d_B, num_bytes);
    cudaMalloc((void**)&d_C, num_bytes);
    cudaMalloc((void**)&d_C1, num_bytes);
    cudaMemset(d_C1, 0, num_bytes);
    cudaMemset(d_A, 0, num_bytes);
    cudaMemset(d_B, 0, num_bytes);
    cudaMemset(d_C, 0, num_bytes);

    //COPY A TO GPU
    cudaMemcpy(d_A, A, num_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, num_bytes, cudaMemcpyHostToDevice);
//call to use global memory
    cudaEvent_t start_event, stop_event;
    float elapsed_time_gpu_g;
    CUDA_SAFE_CALL( cudaEventCreate(&start_event) );
    CUDA_SAFE_CALL( cudaEventCreate(&stop_event) );
    cudaEventRecord(start_event, 0);
    
//    cudaMemcpy(d_A, A, num_bytes, cudaMemcpyHostToDevice);
//    cudaMemcpy(d_B, B, num_bytes, cudaMemcpyHostToDevice);

    dim3 block_size(TILE_WIDTH,TILE_WIDTH);
    dim3 grid_size(ARR_LEN/TILE_WIDTH,ARR_LEN/TILE_WIDTH);   
    kernel_mmm_global<<<grid_size,block_size>>>(d_A,d_B,d_C);

    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    CUDA_SAFE_CALL( cudaEventElapsedTime(&elapsed_time_gpu_g,start_event, stop_event) );

    //COPY B BACK FROM GPU
    cudaMemcpy(h_C, d_C, num_bytes, cudaMemcpyDeviceToHost);
//Call to use shared memory
 cudaEvent_t start_event1, stop_event1;
    float elapsed_time_gpu_s;
    CUDA_SAFE_CALL( cudaEventCreate(&start_event1) );
    CUDA_SAFE_CALL( cudaEventCreate(&stop_event1) );
    cudaEventRecord(start_event1, 0);

    dim3 block_size1(TILE_WIDTH,TILE_WIDTH);
   dim3 grid_size1(ARR_LEN/TILE_WIDTH,ARR_LEN/TILE_WIDTH);
       kernel_mmm_shared<<<grid_size1,block_size1>>>(d_A,d_B,d_C1);


    cudaEventRecord(stop_event1, 0);
    cudaEventSynchronize(stop_event1);
    CUDA_SAFE_CALL( cudaEventElapsedTime(&elapsed_time_gpu_s,start_event1, stop_event1) );

    //COPY B BACK FROM GPU
    cudaMemcpy(h_C1, d_C1, num_bytes, cudaMemcpyDeviceToHost);


    //TODO: Compare results
    for(iter = 0; iter < ARR_LEN; iter++){
       for(i=0; i< ARR_LEN; i ++){
	 if(h_C[iter + i*ARR_LEN] - C[iter][i] > TOL){
          errCount ++;
         }

      }
    }
    //Time compare
    printf("Array Size: %d\n",ARR_LEN);
    printf("Error Count: \t%d\n",errCount);
    printf("CPU computation time: \t%.2f ms\n",elapsed_time_cpu);
    printf("GPU calculation time with global memory:  \t%.2f ms\n", elapsed_time_gpu_g);
    printf("GPU calculation time with shared memory:  \t%.2f ms\n", elapsed_time_gpu_s);
    printf("Shared to Global Ratio:\t%.2fx\n",(elapsed_time_gpu_s/elapsed_time_gpu_g));

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_C1);
    free(h_C);
    free(h_C1);
}
