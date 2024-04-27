#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define SOBEL_KERNEL_SIZE 3
#define TILE_SIZE 32  

__constant__ int sobel_kernel_x[SOBEL_KERNEL_SIZE][SOBEL_KERNEL_SIZE] = {{-1, 0, 1},
                                                                          {-2, 0, 2},
                                                                          {-1, 0, 1}};

__constant__ int sobel_kernel_y[SOBEL_KERNEL_SIZE][SOBEL_KERNEL_SIZE] = {{-1, -2, -1},
                                                                          { 0,  0,  0},
                                                                          { 1,  2,  1}};

__global__ void sobel_filter(unsigned char* input_image, unsigned char* output_image, int width, int height) {
    __shared__ unsigned char shared_image[TILE_SIZE + 2][TILE_SIZE + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * TILE_SIZE + tx;
    int y = blockIdx.y * TILE_SIZE + ty;

    
    if (x < width && y < height) {
        shared_image[ty + 1][tx + 1] = input_image[y * width + x];
    }

    
    if (tx == 0 && x > 0) {
        shared_image[ty + 1][0] = input_image[y * width + x - 1];
    }
    if (tx == TILE_SIZE - 1 && x < width - 1) {
        shared_image[ty + 1][TILE_SIZE + 1] = input_image[y * width + x + 1];
    }
    if (ty == 0 && y > 0) {
        shared_image[0][tx + 1] = input_image[(y - 1) * width + x];
    }
    if (ty == TILE_SIZE - 1 && y < height - 1) {
        shared_image[TILE_SIZE + 1][tx + 1] = input_image[(y + 1) * width + x];
    }

    __syncthreads();

    if (x < width && y < height) {
        int gradient_x = 0;
        int gradient_y = 0;

        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                gradient_x += shared_image[ty + i + 1][tx + j + 1] * sobel_kernel_x[i + 1][j + 1];
                gradient_y += shared_image[ty + i + 1][tx + j + 1] * sobel_kernel_y[i + 1][j + 1];
            }
        }

        output_image[y * width + x] = sqrtf((float)(gradient_x * gradient_x + gradient_y * gradient_y));
    }
}

int main() {
    int width, height, channels;
    unsigned char* input_image = stbi_load("input_image.jpg", &width, &height, &channels, 0);

    if (!input_image) {
        printf("Error loading the image!\n");
        return 1;
    }

    printf("Image loaded successfully: width=%d, height=%d, channels=%d\n", width, height, channels);

    unsigned char* d_input_image;
    unsigned char* d_output_image;

    size_t image_size = width * height;
    size_t image_bytes = image_size * sizeof(unsigned char);

    cudaError_t cudaStatus;

    cudaStatus = cudaMalloc((void**)&d_input_image, image_bytes);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }

    cudaStatus = cudaMalloc((void**)&d_output_image, image_bytes);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_input_image);
        return 1;
    }

    cudaStatus = cudaMemcpy(d_input_image, input_image, image_bytes, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_input_image);
        cudaFree(d_output_image);
        return 1;
    }

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((width + TILE_SIZE - 1) / TILE_SIZE, (height + TILE_SIZE - 1) / TILE_SIZE);

    printf("Launching CUDA kernel with gridDim=(%d, %d) blockDim=(%d, %d)\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y);

    cudaEvent_t start, stop;
    float milliseconds = 0;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    sobel_filter<<<gridDim, blockDim>>>(d_input_image, d_output_image, width, height);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Time taken by CUDA code: %.2f milliseconds\n", milliseconds);

    unsigned char* output_image = (unsigned char*)malloc(image_bytes);

    cudaStatus = cudaMemcpy(output_image, d_output_image, image_bytes, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_input_image);
        cudaFree(d_output_image);
        free(output_image);
        return 1;
    }

    stbi_write_jpg("output_image_cuda.jpg", width, height, 1, output_image, 100);

    printf("Output image saved as output_image_cuda.jpg\n");

    stbi_image_free(input_image);
    free(output_image);

    cudaFree(d_input_image);
    cudaFree(d_output_image);

    return 0;
}
