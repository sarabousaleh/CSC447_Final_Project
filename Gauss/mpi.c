#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

void image_convolution(const unsigned char *input, unsigned char *output, int width, int height, int channels, const float *kernel, int kernel_size)
{
    int pad = kernel_size / 2;
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            for (int c = 0; c < channels; ++c)
            {
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
    }
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    const char *input_file = "input_image.jpg";
    const char *output_file = "output_image_mpi.jpg";
    int width, height, channels;

    unsigned char *image_data = NULL;

    if (world_rank == 0)
    {
        // Load the input image
        image_data = stbi_load(input_file, &width, &height, &channels, 0);
        if (!image_data)
        {
            fprintf(stderr, "Error loading image\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    // Broadcast image dimensions and channels to all processes
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&channels, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (world_rank != 0)
    {
        image_data = malloc(width * height * channels * sizeof(unsigned char));
    }

    // Broadcast image data to all processes
    MPI_Bcast(image_data, width * height * channels, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    float kernel[9] = {
        1 / 9.0f, 1 / 9.0f, 1 / 9.0f,
        1 / 9.0f, 1 / 9.0f, 1 / 9.0f,
        1 / 9.0f, 1 / 9.0f, 1 / 9.0f};

    /*  float kernel[] = {
          1 / 9.0f, 1 / 9.0f, 1 / 9.0f,
          1 / 9.0f, 1 / 9.0f, 1 / 9.0f,
          1 / 9.0f, 1 / 9.0f, 1 / 9.0f};**/

    int kernel_size = 3;

    // unsigned char *output_data = malloc(width * height * channels * sizeof(unsigned char));

    // Divide the work among the processes
    int rows_per_process = height / world_size;
    int remaining_rows = height % world_size;

    int start_row = world_rank * rows_per_process;
    int end_row = start_row + rows_per_process;

    if (world_rank == world_size - 1)
    {
        end_row += remaining_rows;
    }

    int pad_top = 0, pad_bottom = 0;
    if (world_rank == 0)
    {
        pad_top = kernel_size / 2;
    }
    if (world_rank == world_size - 1)
    {
        pad_bottom = kernel_size / 2;
    }

    int padded_width = width + kernel_size - 1;
    int padded_height = end_row - start_row + pad_top + pad_bottom;
    int padded_channels = channels;
    unsigned char *input_data = (unsigned char *)calloc(padded_width * padded_height * padded_channels, sizeof(unsigned char));
    unsigned char *output_data = (unsigned char *)calloc(padded_width * padded_height * padded_channels, sizeof(unsigned char));

    for (int y = 0; y < padded_height; ++y)
    {
        for (int x = 0; x < padded_width; ++x)
        {
            int source_x = x - (kernel_size / 2);
            int source_y = y - pad_top + start_row;
            if (source_x >= 0 && source_x < width && source_y >= 0 && source_y < height)
            {
                for (int c = 0; c < channels; ++c)
                {
                    input_data[(y * padded_width + x) * channels + c] = image_data[(source_y * width + source_x) * channels + c];
                }
            }
        }
    }

    // Perform the convolution for the assigned rows
    int pad = kernel_size / 2; 
    int padded_start_row = start_row == 0 ? start_row : start_row - pad;
    int padded_end_row = end_row == height ? end_row : end_row + pad;

    double start_time = MPI_Wtime();
    image_convolution(image_data + padded_start_row * width * channels, output_data + start_row * width * channels, width, padded_end_row - padded_start_row, channels, kernel, kernel_size);

    for (int y = 0; y < end_row - start_row; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            for (int c = 0; c < channels; ++c)
            {
                image_data[((y + start_row) * width + x) * channels + c] = output_data[((y + pad_top) * padded_width + (x + (kernel_size / 2))) * channels + c];
            }
        }
    }
    // Gather the results from all processes
    int *recvcounts = NULL;
    int *displs = NULL;

    if (world_rank == 0)
    {
        recvcounts = malloc(world_size * sizeof(int));
        displs = malloc(world_size * sizeof(int));

        for (int i = 0; i < world_size; ++i)
        {
            int start = i * rows_per_process;
            int end = start + rows_per_process;
            if (i == world_size - 1)
            {
                end += remaining_rows;
            }
            int padded_start = start == 0 ? start : start - pad;
            int padded_end = end == height ? end : end + pad;

            recvcounts[i] = (padded_end - padded_start) * width * channels;
            displs[i] = padded_start * width * channels;
        }
    }
    unsigned char *gather_output_data = NULL;
    if (world_rank == 0)
    {
        gather_output_data = malloc(width * height * channels * sizeof(unsigned char));
    }

    MPI_Gatherv(output_data + padded_start_row * width * channels, (padded_end_row - padded_start_row) * width * channels, MPI_UNSIGNED_CHAR, gather_output_data, recvcounts, displs, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // MPI_Gatherv(output_data + start_row * width * channels, (end_row - start_row) * width * channels, MPI_UNSIGNED_CHAR, output_data, recvcounts, displs, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    if (world_rank == 0)
    {
        double end_time = MPI_Wtime();
        // Save the output image
        if (!stbi_write_jpg(output_file, width, height, channels, gather_output_data, 100))

        {
            fprintf(stderr, "Error writing output image\n");
            stbi_image_free(image_data);
            free(output_data);
            free(gather_output_data);
            free(recvcounts);
            free(displs);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        
        stbi_image_free(image_data);
        free(output_data);
        free(gather_output_data);
        free(recvcounts);
        free(displs);

        printf("Total time taken: %f seconds\n", end_time - start_time);
    }
    else
    {
        free(image_data);
        free(output_data);
    }

    MPI_Finalize();
    return 0;
}
