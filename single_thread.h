// Optimize this function
#include <immintrin.h>

void singleThread( int input_row, 
                int input_col,
                int *input, 
                int kernel_row, 
                int kernel_col, 
                int *kernel,
                int output_row, 
                int output_col, 
                long long unsigned int *output ) 
/*Given code with unoptimized
{

    for(int i = 0; i < output_row * output_col; ++i)
        output[i] = 0;

    for(int output_i = 0; output_i< output_row; output_i++)
    {
        for(int output_j = 0; output_j< output_col; output_j++)
        {
            for(int kernel_i = 0; kernel_i< kernel_row; kernel_i++)
            {
                for(int kernel_j = 0; kernel_j< kernel_col; kernel_j++)
                {
                    int input_i = (output_i + 2*kernel_i) % input_row;
                    int input_j = (output_j + 2*kernel_j) % input_col;
                    output[output_i * output_col + output_j] += input[input_i*input_col +input_j] * kernel[kernel_i*kernel_col +kernel_j];
                }
            }
        }
    }

}*/
//Method 1:Code Motion + Loop unrolling
/*{

    for(int output_i = 0; output_i< output_row; output_i++)
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

                for(int kernel_j = 0; kernel_j< kernel_col; kernel_j= kernel_j+4)
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
}
*/

//SIMD Optimization
{
    int remain_rows=output_row-kernel_row+1;
    int remain_cols=output_col-kernel_col+1;

    remain_rows=remain_rows-(remain_rows%8);
    remain_cols=remain_cols-(remain_cols%8);

    int * t=(int *)malloc(sizeof(int)*output_row*output_col);
    for(int kernel_i=0;kernel_i<kernel_row;kernel_i++)
    {
        for(int kernel_j=0;kernel_j<kernel_col;kernel_j++)
        {
            __m256i kernel_values=_mm256_set1_epi32(kernel[kernel_i*kernel_col+kernel_j]);
            for(int output_i=0;output_i<output_row;output_i++)
            {
                int input_i=(output_i+2*kernel_i)%input_row;
                int inp_idx=input_i*input_col;
                int otp_idx=output_i*output_col;

                for(int output_j=0;output_j<remain_cols;output_j+=8)
                {
                    int input_j=(output_j+2*kernel_j);
                    __m256i input_values = _mm256_loadu_si256((__m256i *)(input+inp_idx+input_j));
                    __m256i result_values = _mm256_mullo_epi32(kernel_values,input_values);

                    result_values=_mm256_add_epi32(result_values,_mm256_loadu_si256((__m256i *)(t+output_i*output_col+output_j)));
                    _mm256_storeu_si256((__m256i *)(t+output_i*output_col+output_j),result_values);
                }
            }
        }
    }


    for(int i = 0; i < output_row * output_col; ++i)
    output[i] = t[i];

    for(int output_i = 0; output_i< output_row; output_i++)
    {
        int a = output_i * output_col;
        for(int output_j = remain_cols; output_j< output_col; output_j++)
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

                for(int kernel_j = 0; kernel_j< kernel_col; kernel_j= kernel_j+4)
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

    
}