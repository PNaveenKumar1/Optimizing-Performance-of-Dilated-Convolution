
//Ankush Ahirwar

#include<cuda_runtime.h>
#include<cassert>

__global__ void convolved(int input_row,
    int input_col,
    int* input,
    int kernel_row,
    int kernel_col,
    int* kernel,
    int output_row,
    int output_col,
    long long unsigned int* output)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    long long unsigned int temp=0;
    if(row<output_row && col<output_col)
    {
        for (int i = 0; i < kernel_row; i++)
        {
            for (int j = 0; j < kernel_col; j++)
            {
                // Accumulate result
                int input_i = (row + 2 * i ) % input_row;
                int input_j = (col + 2 * j) % input_col;
                temp += input[input_i * input_col + input_j]* kernel[i * kernel_col + j];
            }
        }
        output[row * output_col + col]=temp;
    }
}

void gpuThread(int input_row,
    int input_col,
    int* input,
    int kernel_row,
    int kernel_col,
    int* kernel,
    int output_row,
    int output_col,
    long long unsigned int* output)
{
    //host vector pointers and device vector pointers
    int* h_input = input, * h_kernel = kernel;
    int* d_input, * d_kernel;
    long long unsigned int* h_output = output,*d_output;

    //allocation size this much size required in gpu
    size_t bytes_ip = sizeof(int) * input_row * input_col;
    size_t bytes_ker = sizeof(int) * kernel_row * kernel_col;
    size_t bytes_op = sizeof(long long unsigned int) * output_row * output_col;

    //allocate device memory
    
    cudaMalloc(&d_input, bytes_ip);
    cudaMalloc(&d_kernel, bytes_ker);
    cudaMalloc(&d_output, bytes_op);

    //copy data to device
    cudaMemcpy(d_input, h_input, bytes_ip, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, bytes_ker, cudaMemcpyHostToDevice);

    // Calculate grid dimensions
    int THREADS = 16;
    int BLOCKS_X = (output_row + THREADS - 1) / THREADS;
    int BLOCKS_Y = (output_col + THREADS - 1) / THREADS;
    // Dimension launch arguments
    dim3 block_dim(THREADS, THREADS);
    dim3 grid_dim(BLOCKS_Y, BLOCKS_X);

    // Launch the kernel
    convolved <<<grid_dim, block_dim >>> (input_row,input_col,d_input,kernel_row,kernel_col,d_kernel, output_row,output_col,d_output);
    cudaDeviceSynchronize();
    //cout << "Error: " << cudaGetErrorString(cudaGetLastError()) << '\n';
    // Copy the result from the device to the host
    cudaMemcpy(h_output, d_output, bytes_op, cudaMemcpyDeviceToHost);

    // Free the device memory
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    
}
