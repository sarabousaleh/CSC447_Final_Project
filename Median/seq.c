#include <stdio.h>
#include <stdlib.h>
#include <time.h>  // Include the time.h header for clock()

#define WIDTH 5   // Width of the image
#define HEIGHT 5  // Height of the image

// Function to calculate median
int median(int arr[], int size) {
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

// Applying median filter on an image
void applyMedianFilter(int input[HEIGHT][WIDTH], int output[HEIGHT][WIDTH]) {
    int window[9];  // To hold the values of the 3x3 window

    for (int y = 1; y < HEIGHT-1; y++) {
        for (int x = 1; x < WIDTH-1; x++) {
            // Fill the window with values around the pixel (including the pixel)
            window[0] = input[y-1][x-1];
            window[1] = input[y-1][x];
            window[2] = input[y-1][x+1];
            window[3] = input[y][x-1];
            window[4] = input[y][x];
            window[5] = input[y][x+1];
            window[6] = input[y+1][x-1];
            window[7] = input[y+1][x];
            window[8] = input[y+1][x+1];

            // Calculate the median and assign it to the output pixel
            output[y][x] = median(window, 9);
        }
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

    // Start timing
    clock_t start, end;
    double cpu_time_used;

    start = clock();

    // Apply median filter
    applyMedianFilter(image, filteredImage);

    // Stop timing
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

   
    printf("Filtered Image:\n");
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            printf("%d ", filteredImage[y][x]);
        }
        printf("\n");
    }

    
    printf("Time used: %f seconds\n", cpu_time_used);

    return 0;
}

