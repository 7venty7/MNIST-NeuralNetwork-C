#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "utils/utils.h"

int main(void) {
  const double learning_rate = 0.1f;

  // loading data
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

  data training_data[BATCH_SIZE];

  load_training_data(training_data);

  // initialising arrays
  double hidden_layer_1[N_NODES];
  double hidden_layer_2[N_NODES];
  double output_layer[N_OUTPUT];

  double input_layer_bias[N_NODES];
  double hidden_layer_bias[N_NODES];
  double output_layer_bias[N_OUTPUT];

  double input_layer_weights[N_NODES][INPUTSIZE];
  double hidden_layer_weights[N_NODES][N_NODES];
  double output_layer_weights[N_OUTPUT][N_NODES];

  // randomising initial weights
  for (int i = 0; i < N_NODES; i++) {
    for (int j = 0; j < INPUTSIZE; j++) {
      input_layer_weights[i][j] = init_weights();
    }
  }

  for (int i = 0; i < N_NODES; i++) {
    for (int j = 0; j < N_NODES; j++) {
      hidden_layer_weights[i][j] = init_weights();
    }
  }

  for (int i = 0; i < N_OUTPUT; i++) {
    for (int j = 0; j < N_NODES; j++) {
      output_layer_weights[i][j] = init_weights();
    }
  }

  // randomising initial biases
  for (int i = 0; i < N_NODES; i++) {
    input_layer_bias[i] = init_weights();
  }

  for (int i = 0; i < N_NODES; i++) {
    hidden_layer_bias[i] = init_weights();
  }

  for (int i = 0; i < N_OUTPUT; i++) {
    output_layer_bias[i] = init_weights();
  }

  fclose(training_images);
  fclose(training_labels);
}
