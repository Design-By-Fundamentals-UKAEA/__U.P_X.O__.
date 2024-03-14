#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <input_filename>\n", argv[0]);
        return EXIT_FAILURE;
    }

    char *input_filename = argv[1];
    char output_filename[256];

    // Create an output filename by appending "_updated" before the file extension
    strcpy(output_filename, input_filename);
    char *dot = strrchr(output_filename, '.');
    if (dot != NULL) {
        strcpy(dot, "_updated.bin");
    } else {
        strcat(output_filename, "_updated.bin");
    }

    // Open the input file
    FILE *file_in = fopen(input_filename, "rb");
    if (file_in == NULL) {
        perror("Unable to open input file");
        return EXIT_FAILURE;
    }

    // Read the shape of the array
    int shape[3];
    fread(shape, sizeof(int), 3, file_in);

    // Allocate memory for the array
    int size = shape[0] * shape[1] * shape[2];
    int *array = (int*)malloc(size * sizeof(int));
    if (array == NULL) {
        perror("Memory allocation failed");
        fclose(file_in);
        return EXIT_FAILURE;
    }

    // Read the array data
    fread(array, sizeof(int), size, file_in);
    fclose(file_in);

    // Update the array
    for (int i = 0; i < size; i++) {
        array[i] += 1;
    }

    // Write the updated array to the output file
    FILE *file_out = fopen(output_filename, "wb");
    if (file_out == NULL) {
        perror("Unable to open output file");
        free(array);
        return EXIT_FAILURE;
    }

    // Optionally write the shape of the array
    fwrite(shape, sizeof(int), 3, file_out);
    // Write the updated array data
    fwrite(array, sizeof(int), size, file_out);

    fclose(file_out);
    free(array);

    printf("Updated array written to %s\n", output_filename);

    return EXIT_SUCCESS;
}
