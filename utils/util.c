#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
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

double init_weight() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  srand(tv.tv_usec * tv.tv_sec);

  return ((double) rand() / (double) RAND_MAX);
}

double relu(double x) {
  if (x < 0) {
    return 0;
  }

  return x;
}

double d_relu(double x) {
  if (x > 0) {
    return 1;
  }

  return 0;
}

void shuffle(double *arr, size_t len) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  srand48(tv.tv_usec * tv.tv_sec);

  if (len > 1) {
    for (int i = len - 1; i >= 0; i--) {
      int j = (unsigned int) (drand48() * (i + 1));
      double tmp = arr[j];
      arr[j] = arr[i];
      arr[i] = tmp;
    }
  }
}

void load_training_data(data *training_data, FILE *training_images, FILE *training_labels) {
  static unsigned long int offset = 0;
  int index = 0;

  fseek(training_images, 16 + (offset * INPUTSIZE), SEEK_CUR);
  fseek(training_labels, 8 + offset, SEEK_CUR);

  while (index < BATCH_SIZE) {
    fread(training_data[index].image, sizeof(uint8_t), INPUTSIZE, training_images);
    fread(&training_data[index].label, sizeof(uint8_t), 1, training_labels);

    index++;
  }

  offset++;
  offset %= TRAINING_SETS;
}
