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

  training_images = fopen("train-images.idx3-ubyte", "rb");
  training_labels = fopen("train-labels.idx1-ubyte", "rb");

  if (training_images == NULL || training_labels == NULL) {
    printf("Error loading files\n");
    exit(1);
  }

  uint32_t magic_n1;
  uint32_t magic_n2;

  fread(&magic_n1, sizeof(uint32_t), 1, training_images);
  fread(&magic_n2, sizeof(uint32_t), 1, training_labels);

  printf("Magic number of training image file: %d\n", magic_n1);
  printf("Magic number of training label file: %d\n", magic_n2);

  fseek(training_images, 16, SEEK_CUR);
  fseek(training_labels, 8, SEEK_CUR);

  int index = 0;
  uint8_t label = 0;

  while (index < TRAINING_SETS) {
    fread(training_data[index].image, sizeof(uint8_t), INPUTSIZE, training_images);
    fread(&training_data[index].label, sizeof(uint8_t), 1, training_labels);

    index++;
  }

  fclose(training_images);
  fclose(training_labels);
}
