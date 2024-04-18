#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <cuda_runtime.h>

__global__ void image_convolution_kernel(const unsigned char *input, unsigned char *output, int width, int height, int channels, const float *kernel, int kernel_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;

    if (x < width && y < height && c < channels)
    {
        int pad = kernel_size / 2;
        float sum = 0;

        for (int ky = -pad; ky <= pad; ++ky)
        {
            for (int kx = -pad; kx <= pad; ++kx)
            {
                int ix = x + kx;
                int iy = y + ky;

                if (ix >= 0 && ix < width && iy >= 0 && iy < height)
                {
                    float pixel = input[(iy * width + ix) * channels + c] / 255.0f;
                    float kernel_val = kernel[(ky + pad) * kernel_size + (kx + pad)];
                    sum += pixel * kernel_val;
                }
            }
        }
        output[(y * width + x) * channels + c] = (unsigned char)(sum * 255);
    }
}

int main(int argc, char *argv[])
{
    const char *input_file = "input_image.jpg";
    const char *output_file = "output_image_cuda.jpg";
    int width, height, channels;

    unsigned char *image_data = stbi_load(input_file, &width, &height, &channels, 0);
    if (!image_data)
    {
        fprintf(stderr, "Error loading image\n");
        return 1;
    }

    float kernel[] = {
        -2, -1, 0,
        -1, 1, 1,
        0, 1, 2};
    int kernel_size = 3;

    unsigned char *d_input, *d_output;
    float *d_kernel;

    size_t image_size = width * height * channels * sizeof(unsigned char);

    cudaMalloc((void **)&d_input, image_size);
    cudaMalloc((void **)&d_output, image_size);
    cudaMalloc((void **)&d_kernel, kernel_size * kernel_size * sizeof(float));

    cudaMemcpy(d_input, image_data, image_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16, 1);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   channels);

    // Timing setup
    cudaEvent_t start, stop;
    float milliseconds = 0;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start time
    cudaEventRecord(start);

    image_convolution_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, width, height, channels, d_kernel, kernel_size);

    // Record end time
    cudaEventRecord(stop);

    // Synchronize events
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Time taken: %f ms\n", milliseconds);

    cudaMemcpy(image_data, d_output, image_size, cudaMemcpyDeviceToHost);

    stbi_write_jpg(output_file, width, height, channels, image_data, 100);

    stbi_image_free(image_data);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);

    return 0;
}
