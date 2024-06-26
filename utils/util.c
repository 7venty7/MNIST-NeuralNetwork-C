#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utils.h"

void matrix_vector_mult(double **weights, double *activations, double *nxt_layer, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    nxt_layer[i] = 0;

    for (int j = 0; j < cols; j++) {
      nxt_layer[i] += weights[i][j] * activations[j];
    }
  }
}

void vec_add(double *activations, double *biases, int len) {
  for (int i = 0; i < len; i++) {
    activations[i] += biases[i];
  }
}

void load_training_data(data *training_data) {
  FILE *training_images;
  FILE *training_labels;

  training_images = fopen("../training_images", "rb");
  training_labels = fopen("../training_labels", "rb");

  if (training_images == NULL || training_data == NULL) {
    printf("Error loading files\n");
    return;
  }

  uint32_t magic_n1;
  uint32_t magic_n2;

  fread(&magic_n1, sizeof(uint32_t), 1, training_images);
  fread(&magic_n2, sizeof(uint32_t), 1, training_labels);

  printf("Magic number of training image file: %d\n", magic_n1);
  printf("Magic number of training label file: %d\n", magic_n2);

  fclose(training_images);
  fclose(training_labels);
}
