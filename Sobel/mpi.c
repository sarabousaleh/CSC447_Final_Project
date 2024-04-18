#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define SOBEL_KERNEL_SIZE 3

int sobel_kernel_x[SOBEL_KERNEL_SIZE][SOBEL_KERNEL_SIZE] = {{-1, 0, 1},
                                                            {-2, 0, 2},
                                                            {-1, 0, 1}};

int sobel_kernel_y[SOBEL_KERNEL_SIZE][SOBEL_KERNEL_SIZE] = {{-1, -2, -1},
                                                            { 0,  0,  0},
                                                            { 1,  2,  1}};

void sobel_filter(unsigned char* local_input, unsigned char* local_output, int local_height, int width, int height, int rank) {
    for (int y = 1; y < local_height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int gradient_x = 0;
            int gradient_y = 0;
            
            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    gradient_x += local_input[(y + i) * width + (x + j)] * sobel_kernel_x[i + 1][j + 1];
                    gradient_y += local_input[(y + i) * width + (x + j)] * sobel_kernel_y[i + 1][j + 1];
                }
            }
            
            local_output[y * width + x] = sqrt(gradient_x * gradient_x + gradient_y * gradient_y);
        }
    }
}

int main(int argc, char* argv[]) {
    int rank, size;
    int width, height, channels;
    unsigned char* input_image;
    unsigned char* local_input;
    unsigned char* local_output;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0) {
        input_image = stbi_load("input_image.jpg", &width, &height, &channels, 0);
        if (!input_image) {
            printf("Error loading the image!\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    int local_height = height / size;
    local_input = (unsigned char*)malloc(width * local_height);
    local_output = (unsigned char*)malloc(width * local_height);
    
    MPI_Scatter(input_image, width * local_height, MPI_UNSIGNED_CHAR, local_input, width * local_height, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    
    double start_time = MPI_Wtime();
    
    sobel_filter(local_input, local_output, local_height, width, height, rank);
    
    double end_time = MPI_Wtime();
    
    if (rank == 0) {
        printf("Time taken for MPI Sobel filter: %f seconds\n", end_time - start_time);
    }
    
    unsigned char* output_image = NULL;
    
    if (rank == 0) {
        output_image = (unsigned char*)malloc(width * height);
    }
    
    MPI_Gather(local_output, width * local_height, MPI_UNSIGNED_CHAR, output_image, width * local_height, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        stbi_write_jpg("output_image_mpi.jpg", width, height, 1, output_image, 100);
        stbi_image_free(input_image);
        free(output_image);
    }
    
    free(local_input);
    free(local_output);
    
    MPI_Finalize();
    
    return 0;
}
