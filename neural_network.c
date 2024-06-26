#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "utils/utils.h"

int main(void) {
  printf("Hello\n");
  const double learning_rate = 0.1f;

  // loading data
  printf("Hello\n");
  data training_data[TRAINING_SETS];
  printf("Hello\n");

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


}
