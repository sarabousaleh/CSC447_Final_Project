#include <stdio.h>
#include <stdlib.h>

#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

void image_convolution(const float *input, float *output, int width, int height, const float *kernel, int kernel_size)
{
    int pad = kernel_size / 2;
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
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
                        float pixel = input[iy * width + ix];
                        float kernel_val = kernel[(ky + pad) * kernel_size + (kx + pad)];
                        sum += pixel * kernel_val;
                    }
                }
            }
            output[y * width + x] = sum;
        }
    }
}

int main()
{
    const char *input_file = "input_image.jpg";
    const char *output_file = "output_image_seq.jpg";
    int width, height, channels;

    // Load the input image
    unsigned char *image_data = stbi_load(input_file, &width, &height, &channels, 0);
    if (!image_data)
    {
        fprintf(stderr, "Error loading image\n");
        return 1;
    }

    // Convert the input image to grayscale
    float *grayscale_data = malloc(width * height * sizeof(float));
    for (int i = 0; i < width * height; ++i)
    {
        float r = image_data[i * channels + 0] / 255.0f;
        float g = image_data[i * channels + 1] / 255.0f;
        float b = image_data[i * channels + 2] / 255.0f;
        grayscale_data[i] = 0.299f * r + 0.587f * g + 0.114f * b;
    }

    // Define the convolution kernel 
    float kernel[9] = {
        1 / 9.0f, 1 / 9.0f, 1 / 9.0f,
        1 / 9.0f, 1 / 9.0f, 1 / 9.0f,
        1 / 9.0f, 1 / 9.0f, 1 / 9.0f};

    int kernel_size = 3;
    /*  1 / 16.0f, 2 / 16.0f, 1 / 16.0f,
      2 / 16.0f, 4 / 16.0f, 2 / 16.0f,
      1 / 16.0f, 2 / 16.0f, 1 / 16.0f};*/

    // Perform the convolution
    float *convoluted_data = malloc(width * height * sizeof(float));
    clock_t start_time = clock();
    image_convolution(grayscale_data, convoluted_data, width, height, kernel, kernel_size);
    clock_t end_time = clock(); // Measure the end time

    double elapsed_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC; // Calculate the elapsed time in seconds
    printf("Elapsed time: %f seconds\n", elapsed_time);                       // Print the elapsed time

    // Convert the convoluted image back to RGB format
    unsigned char *output_data = malloc(width * height * channels * sizeof(unsigned char));
    for (int i = 0; i < width * height; ++i)
    {
        output_data[i * channels + 0] = convoluted_data[i] * image_data[i * channels + 0];
        output_data[i * channels + 1] = convoluted_data[i] * image_data[i * channels + 1];
        output_data[i * channels + 2] = convoluted_data[i] * image_data[i * channels + 2];
        if (channels == 4)
        {
            output_data[i * channels + 3] = image_data[i * channels + 3]; // Preserve the alpha channel if it exists
        }
    }

    // Save the output image
    if (!stbi_write_jpg(output_file, width, height, channels, output_data, 100))
    {
        fprintf(stderr, "Error writing output image\n");
        stbi_image_free(image_data);
        free(grayscale_data);
        free(convoluted_data);
        free(output_data);
        return 1;
    }

    
    stbi_image_free(image_data);
    free(grayscale_data);
    free(convoluted_data);
    free(output_data);

    return 0;
}
