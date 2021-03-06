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

__global__ void kernel_sor(float *d_A, float *d_B) {
    const int row_thread_x = ARR_LEN/blockDim.x;
    const int row_thread_y = ARR_LEN/blockDim.y;
    const int threadStart = blockIdx.x * row_thread_x;
    const int threadEnd = (blockIdx.x+1)*row_thread_x -1;
    const int width = threadStart - threadEnd;
    int index,iter;
    index = row_thread_x*width+row_thread_y; 
    // checkboundaries
    for(iter =1; iter <LOOP; iter++){
if((row_thread_x > 0) && (row_thread_y > 0) && (row_thread_x < ARR_LEN-1) && (row_thread_y < ARR_LEN-1))
        d_B[index] = d_A[index-1]
		   + d_A[index+1]
		   + d_A[index+ARR_LEN]
		   + d_A[index-ARR_LEN];
	__syncthreads();   
	}

    }

main (int argc, char **argv) {

    float A[ARR_LEN][ARR_LEN], B[ARR_LEN][ARR_LEN];
    float *d_A, *d_B;
    float *h_B; // output of B from the GPU to CPU
    int i, j,iter;
    int num_bytes = ARR_LEN * ARR_LEN * sizeof(float);
    int errCount = 0;
    // Input is randomly generated
	for(i=0;i<ARR_LEN;i++) {
             for(j=0;j<ARR_LEN;j++) {
           	 A[i][j] = (float) rand()/1234;
            
       	      }
         }
    

    cudaEvent_t start_event0, stop_event0;
    float elapsed_time0;
    CUDA_SAFE_CALL( cudaEventCreate(&start_event0) );
    CUDA_SAFE_CALL( cudaEventCreate(&stop_event0) );
    cudaEventRecord(start_event0, 0);
    // CPU computation
   for(iter=1;iter<LOOP;iter++){
      for(i=1;i<ARR_LEN-1;i++) {
        for(j=1;j<ARR_LEN-1;j++) {
            B[i][j] = A[i-1][j]+A[i+1][j]+A[i][j-1]+A[i][j+1];
        }
      }
    }
    cudaEventRecord(stop_event0, 0);
    cudaEventSynchronize(stop_event0);
    CUDA_SAFE_CALL( cudaEventElapsedTime(&elapsed_time0,start_event0, stop_event0) );



    h_B = (float *)malloc(num_bytes);
    memset(h_B, 0, num_bytes);
    //ALLOCATE MEMORY FOR GPU COPIES OF A AND B
    cudaMalloc((void**)&d_A, num_bytes);
    cudaMalloc((void**)&d_B, num_bytes);
    cudaMemset(d_A, 0, num_bytes);
    cudaMemset(d_B, 0, num_bytes);

    //COPY A TO GPU
    cudaMemcpy(d_A, A, num_bytes, cudaMemcpyHostToDevice);

    // create CUDA event handles for timing purposes
    cudaEvent_t start_event, stop_event;
    float elapsed_time;
    CUDA_SAFE_CALL( cudaEventCreate(&start_event) );
    CUDA_SAFE_CALL( cudaEventCreate(&stop_event) );
    cudaEventRecord(start_event, 0);

    dim3 block_size(16,16); //values experimentally determined to be fastes
    
    kernel_sor<<<1,block_size>>>(d_A,d_B); 

    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    CUDA_SAFE_CALL( cudaEventElapsedTime(&elapsed_time,start_event, stop_event) );

    //COPY B BACK FROM GPU
    cudaMemcpy(h_B, d_B, num_bytes, cudaMemcpyDeviceToHost);
     //TODO: Compare results
    int index = iter + i*ARR_LEN;
    for(iter = 0; iter < ARR_LEN; iter++){
       for(i=0; i< ARR_LEN; i ++){
         if(h_B[index] - B[iter][i] > TOL){
          errCount ++;
         }
      }
    }
    printf("Error Count:\t%d\n",errCount);
    printf("CPU computation time:  \t%.2f ms\n", elapsed_time0);
    printf("GPU computation time:  \t%.2f ms\n", elapsed_time);
    printf("CUDA Speedup:\t%.2fx\n",(elapsed_time0/elapsed_time));

    cudaFree(d_A);
    cudaFree(d_B);
    free(h_B);
}
