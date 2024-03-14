#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <input_filename> <output_filename>\n", argv[0]);
        return 1;
    }

    FILE *file_in, *file_out;
    int num;
    char line[1024]; // Buffer for reading lines; adjust size as needed

    // Open the input file for reading
    file_in = fopen(argv[1], "r");
    if (file_in == NULL) {
        perror("Error opening input file");
        return 1;
    }

    // Open the output file for writing
    file_out = fopen(argv[2], "w");
    if (file_out == NULL) {
        perror("Error opening output file");
        fclose(file_in); // Make sure to close the input file before returning
        return 1;
    }

    // Read each line from the file
    while (fgets(line, sizeof(line), file_in) != NULL) {
        char *token = strtok(line, " \n"); // Split the line into tokens based on space and newline characters

        while (token != NULL) {
            num = atoi(token) + 1; // Convert token to integer and add 1
            fprintf(file_out, "%d ", num); // Write the updated number to the output file
            token = strtok(NULL, " \n"); // Continue tokenizing the same line
        }
        fprintf(file_out, "\n"); // Write a newline character after processing each row
    }

    // Close the files
    fclose(file_in);
    fclose(file_out);

    return 0;
}
