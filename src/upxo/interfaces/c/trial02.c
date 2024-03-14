#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <input_filename> <output_filename>\n", argv[0]);
        return 1;
    }

    FILE *file_in, *file_out;
    int num;

    file_in = fopen(argv[1], "r");
    if (file_in == NULL) {
        perror("Error opening input file");
        return 1;
    }

    file_out = fopen(argv[2], "w");
    if (file_out == NULL) {
        perror("Error opening output file");
        fclose(file_in);
        return 1;
    }

    while (fscanf(file_in, "%d", &num) == 1) {
        num += 1; // Add 1 to each element
        fprintf(file_out, "%d\n", num); // Write the modified number to the output file
    }

    fclose(file_in);
    fclose(file_out);

    return 0;
}
