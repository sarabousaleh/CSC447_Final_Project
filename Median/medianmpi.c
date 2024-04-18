#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MEDIAN_KERNEL_SIZE 3
#define MEDIAN_KERNEL_HALF_SIZE (MEDIAN_KERNEL_SIZE / 2)

int compare(const void *a, const void *b) {
    return (*(unsigned char*)a - *(unsigned char*)b);
}

void median_filter_channel(unsigned char* input_channel, unsigned char* output_channel, int width, int height, int start_row, int end_row) {
    for (int y = start_row; y < end_row; y++) {
        for (int x = MEDIAN_KERNEL_HALF_SIZE; x < width - MEDIAN_KERNEL_HALF_SIZE; x++) {
            unsigned char window[MEDIAN_KERNEL_SIZE * MEDIAN_KERNEL_SIZE];
            int k = 0;
            for (int i = -MEDIAN_KERNEL_HALF_SIZE; i <= MEDIAN_KERNEL_HALF_SIZE; i++) {
                for (int j = -MEDIAN_KERNEL_HALF_SIZE; j <= MEDIAN_KERNEL_HALF_SIZE; j++) {
                    int idx = (y + i) * width + (x + j);
                    if (idx >= 0 && idx < width * height)  // Boundary check
                        window[k++] = input_channel[idx];
                }
            }
            qsort(window, k, sizeof(unsigned char), compare);
            output_channel[y * width + x] = window[k / 2];
        }
    }
}

int main(int argc, char **argv) {
    int rank, size, width = 1024, height = 768, channels = 3;  // Example dimensions
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int *sendcounts = malloc(sizeof(int) * size);
    int *displs = malloc(sizeof(int) * size);

    // Calculate send counts and displacements
    int sum = 0;
    for (int i = 0; i < size; i++) {
        sendcounts[i] = (width * channels * (height/size)) + (i < height%size ? width * channels : 0);
        displs[i] = sum;
        sum += sendcounts[i];
    }

    unsigned char *input_image = NULL, *output_image = NULL;
    if (rank == 0) {
        input_image = (unsigned char *)malloc(sum);  // Sum is total number of elements after last displacement
        output_image = (unsigned char *)malloc(sum);
        memset(input_image, 128, sum);  // Dummy data initialization
    }

    unsigned char *local_input_image = (unsigned char *)malloc(sendcounts[rank]);
    unsigned char *local_output_image = (unsigned char *)malloc(sendcounts[rank]);

    // Broadcast the image dimensions to all processes
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&channels, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Start the timer
    double start_time = MPI_Wtime();

    // Scatter the input image across all processes
    MPI_Scatterv(input_image, sendcounts, displs, MPI_UNSIGNED_CHAR, 
                 local_input_image, sendcounts[rank], MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // Apply the median filter
    median_filter_channel(local_input_image, local_output_image, width, sendcounts[rank]/(width * channels), 0, sendcounts[rank]/(width * channels));

    // Gather the filtered parts of the image back to the root process
    MPI_Gatherv(local_output_image, sendcounts[rank], MPI_UNSIGNED_CHAR, output_image, sendcounts, displs, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // Stop the timer
    double end_time = MPI_Wtime();

    // Calculate the elapsed time
    double elapsed_time = end_time - start_time;

    if (rank == 0) {
        printf("Processing time: %f seconds\n", elapsed_time);
        // Here you would save or process your output image
        // Save image code would go here...
        free(output_image);
        free(input_image);
    }

    free(local_input_image);
    free(local_output_image);
    free(sendcounts);
    free(displs);

    MPI_Finalize();
    return 0;
}

