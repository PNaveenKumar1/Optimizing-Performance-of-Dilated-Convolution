
#include <pthread.h>
#include <immintrin.h>
// Create other necessary functions here
struct arg_struct {
    int input_row;
    int input_col;
    int *input;
    int kernel_row;
    int kernel_col;
    int *kernel;
    int output_row;
    int output_col;
    long long unsigned int *output;
    int output_start_row;
    int output_end_row;
};

// Fill in this function
void *operation(void *args)
{
    struct arg_struct *data = (struct arg_struct *)args;

    int input_row = data->input_row; 
    int input_col = data->input_col;
    int *input = data->input;
    int kernel_row = data->kernel_row; 
    int kernel_col = data->kernel_col; 
    int *kernel = data->kernel;
    int output_row = data->output_row; 
    int output_col = data->output_col;
    long long unsigned int *output = data->output;
    int output_start_row = data->output_start_row;
    int output_end_row = data->output_end_row;


    for (int output_i = output_start_row; output_i <= output_end_row; output_i++)
    {
        int a = output_i * output_col;
        for(int output_j = 0; output_j< output_col; output_j++)
        {
            int out_idx = a + output_j;
            long long unsigned int x = 0;
            for(int kernel_i = 0; kernel_i< kernel_row; kernel_i++)
            {
                int input_i = output_i + (kernel_i)*2;
                if(input_i >= input_row)
                {
                    input_i -= input_row;
                }
                int b = input_i*input_col;
                int c = kernel_i*kernel_col;

                for(int kernel_j = 0; kernel_j< kernel_col; kernel_j = kernel_j + 4)
                {
                    int input_j = output_j + (kernel_j)*2;
                    if(input_j >= input_col)
                    {
                        input_j -= input_col;
                    }
                    x += input[b +input_j] * kernel[c + kernel_j];
                    if(kernel_j + 1 < kernel_col)
                    {
                        input_j = output_j + 2*(kernel_j+1);
                        if(input_j >= input_col)
                        {
                            input_j -= input_col;
                        }
                        x += input[b + input_j] * kernel[c + kernel_j+1];
                    }
                    if(kernel_j + 2 < kernel_col)
                    {
                        input_j = output_j + 2*(kernel_j+2);
                        if(input_j >= input_col)
                        {
                            input_j -= input_col;
                        }
                        x += input[b + input_j] * kernel[c + kernel_j+2];
                    }
                    if(kernel_j + 3 < kernel_col)
                    {
                        input_j = output_j + 2*(kernel_j+3);
                        if(input_j >= input_col)
                        {
                            input_j -= input_col;
                        }
                        x += input[b + input_j] * kernel[c + kernel_j+3];
                    }
                }
            }
            output[out_idx] = x;
        }
    }
    pthread_exit(NULL);
}

void multiThread( int input_row, 
                int input_col,
                int *input, 
                int kernel_row, 
                int kernel_col, 
                int *kernel,
                int output_row, 
                int output_col, 
                long long unsigned int *output ) 
{
    int Number_of_Threads = 32 ;
    pthread_t threads[Number_of_Threads];

    struct arg_struct *args = (struct arg_struct *)malloc(Number_of_Threads * sizeof(struct arg_struct));

    int threadrows = output_col/Number_of_Threads;

    for (int i = 0; i < Number_of_Threads; i++)
    {
        args[i].input_row = input_row;
        args[i].input_col = input_col;
        args[i].input = input;
        args[i].kernel_row = kernel_row;
        args[i].kernel_col = kernel_col;
        args[i].kernel = kernel;
        args[i].output_row = output_row;
        args[i].output_col = output_col;
        args[i].output = output;

        args[i].output_start_row = i*threadrows;
        if(i == 31)
            args[i].output_end_row = output_row - 1;
        else
            args[i].output_end_row = (i + 1)*threadrows-1;
        
        pthread_create(&threads[i], NULL, operation, (void *)(&args[i]));

    }

    for (int i = 0; i < Number_of_Threads; i++)
        pthread_join(threads[i], NULL);
    
    free(args);
}

