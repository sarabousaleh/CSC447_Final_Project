#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define MEDIAN_KERNEL_SIZE 3
#define MEDIAN_KERNEL_HALF_SIZE (MEDIAN_KERNEL_SIZE / 2)


int compare(const void * a, const void * b) {
   return (*(unsigned char*)a - *(unsigned char*)b);
}

void median_filter_channel(unsigned char* input_channel, unsigned char* output_channel, int width, int height) {
    #pragma omp parallel for collapse(2)
    for (int y = MEDIAN_KERNEL_HALF_SIZE; y < height - MEDIAN_KERNEL_HALF_SIZE; y++) {
        for (int x = MEDIAN_KERNEL_HALF_SIZE; x < width - MEDIAN_KERNEL_HALF_SIZE; x++) {
            unsigned char window[MEDIAN_KERNEL_SIZE * MEDIAN_KERNEL_SIZE];
            int k = 0;
            // Collect all pixels in the window into an array
            for (int i = -MEDIAN_KERNEL_HALF_SIZE; i <= MEDIAN_KERNEL_HALF_SIZE; i++) {
                for (int j = -MEDIAN_KERNEL_HALF_SIZE; j <= MEDIAN_KERNEL_HALF_SIZE; j++) {
                    window[k++] = input_channel[(y + i) * width + (x + j)];
                }
            }
            // Sort the array to find the median
            qsort(window, k, sizeof(unsigned char), compare);
            // Set the median value to the output image
            output_channel[y * width + x] = window[k / 2];
        }
    }
}

int main() {
    int width, height, channels;
    // Replace with the path to your input image
    unsigned char *input_image = stbi_load("input_image.jpg", &width, &height, &channels, 0);

    if (!input_image) {
        printf("Error loading the image!\n");
        return 1;
    }

    size_t img_size = width * height * channels;
    unsigned char *output_image = (unsigned char*)malloc(img_size);
    memcpy(output_image, input_image, img_size); // Copy input to output

    // Start the timer
    double start_time = omp_get_wtime();

    // Apply median filter to each channel separately
    unsigned char *channel_buffer_in = (unsigned char*)malloc(width * height);
    unsigned char *channel_buffer_out = (unsigned char*)malloc(width * height);

    for (int c = 0; c < channels; c++) {
        // Extract one channel
        for (int i = 0; i < width * height; i++) {
            channel_buffer_in[i] = input_image[i * channels + c];
        }

        // Apply the median filter to the single channel
        median_filter_channel(channel_buffer_in, channel_buffer_out, width, height);

        // Merge back the processed channel
        for (int i = 0; i < width * height; i++) {
            output_image[i * channels + c] = channel_buffer_out[i];
        }
    }

    
    double end_time = omp_get_wtime();

    
    stbi_write_jpg("output_image.jpg", width, height, channels, output_image, 100);

    
    printf("Time taken for median filtering: %f seconds\n", end_time - start_time);

    
    stbi_image_free(input_image);
    free(output_image);
    free(channel_buffer_in);
    free(channel_buffer_out);

    return 0;
}

