#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define WIDTH 5   // Width of the image
#define HEIGHT 5  // Height of the image
#define TILE_WIDTH 3  // Tile width for shared memory

// Function to calculate median
__device__ int median(int arr[], int size) {
    for (int i = 0; i < size-1; i++) {
        for (int j = 0; j < size-i-1; j++) {
            if (arr[j] > arr[j+1]) {
                int temp = arr[j];
                arr[j] = arr[j+1];
                arr[j+1] = temp;
            }
        }
    }
    return arr[size/2];
}

// Kernel to apply median filter
__global__ void applyMedianFilterKernel(int *input, int *output, int width, int height) {
    __shared__ int tile[TILE_WIDTH*TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * TILE_WIDTH;
    int by = blockIdx.y * TILE_WIDTH;
    
    int x = bx + tx;
    int y = by + ty;

    // Copy data to shared memory
    if (x < width && y < height) {
        tile[ty * TILE_WIDTH + tx] = input[y * width + x];
    }
    __syncthreads();

    // Apply median filter to the tile
    if (x < width && y < height) {
        int window[9];
        int k = 0;
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                int nx = x + j;
                int ny = y + i;
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    window[k++] = tile[(ty+i) * TILE_WIDTH + (tx+j)];
                }
            }
        }
        output[y * width + x] = median(window, k);
    }
}

int main() {
    int image[HEIGHT][WIDTH] = {
        {10, 20, 30, 40, 50},
        {10, 20, 30, 40, 50},
        {10, 20, 90, 80, 70},
        {10, 20, 30, 40, 50},
        {10, 20, 30, 40, 50}
    };
    int filteredImage[HEIGHT][WIDTH];
    int *d_input, *d_output;

    // Allocate device memory
    cudaMalloc((void**)&d_input, WIDTH * HEIGHT * sizeof(int));
    cudaMalloc((void**)&d_output, WIDTH * HEIGHT * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_input, image, WIDTH * HEIGHT * sizeof(int), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((WIDTH + TILE_WIDTH - 1) / TILE_WIDTH, (HEIGHT + TILE_WIDTH - 1) / TILE_WIDTH);

    // Start timing
    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Launch kernel
    applyMedianFilterKernel<<<gridDim, blockDim>>>(d_input, d_output, WIDTH, HEIGHT);

    // Stop timing
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // Print the time used
    printf("Time used: %f ms\n", elapsedTime);

    // Copy result back to host
    cudaMemcpy(filteredImage, d_output, WIDTH * HEIGHT * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the filtered image
    printf("Filtered Image:\n");
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            printf("%d ", filteredImage[y][x]);
        }
        printf("\n");
    }

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
