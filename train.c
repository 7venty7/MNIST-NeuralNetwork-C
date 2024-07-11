#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include "matrix/matrix.h"
#include "nnet/nnet.h"

int main(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  srand(tv.tv_usec * tv.tv_sec);

  const double learning_rate = 0.02f;

  // load data from files

  FILE *training_images;
  FILE *training_labels;

  training_images = fopen("train-images.idx3-ubyte", "rb");
  training_labels = fopen("train-labels.idx1-ubyte", "rb");

  if (training_images == NULL || training_labels == NULL) {
    printf("Error loading files\n");
    exit(1);
  }

  fseek(training_images, 16, SEEK_SET);
  fseek(training_labels, 8, SEEK_SET);

  input_data training_data;

  // initialise weights and layers

  Matrix *hidden_layer_weights = matrix_init(N_NODES, INPUTSIZE);
  Matrix *output_layer_weights = matrix_init(N_OUTPUT, N_NODES);


  randomise2(hidden_layer_weights, N_NODES);
  randomise2(output_layer_weights, N_OUTPUT);

  for (int i = 0; i < N_IMAGES; i++) {
    // initial forward pass
    Matrix *input_layer = matrix_init(INPUTSIZE, 1);
    load_training_data(&training_data, training_images, training_labels);

    for (int j = 0; j < INPUTSIZE; j++) {
      input_layer->values[j][0] = (double) training_data.image[j] / 256;
    }

    // map(input_layer, sigmoid);

    Matrix *hidden_layer = matrix_mult(hidden_layer_weights, input_layer);
    map(hidden_layer, sigmoid);

    Matrix *output_layer = matrix_mult(output_layer_weights, hidden_layer);
    map(output_layer, sigmoid);

    Matrix *softmaxxed = softmax(output_layer);
    int prediction = max(softmaxxed);
    printf("Label: %d, Prediction: %d\n", training_data.label, prediction);

    for (int j = 0; j < N_OUTPUT; j++) {
      printf("%.4f ", output_layer->values[j][0]);
    }
    printf("\n");

    Matrix *label = matrix_init(N_OUTPUT, 1);
    label->values[training_data.label][0] = 1;

    // backprop

    // compute errors

    Matrix *output_error = matrix_subtract(label, output_layer);

    Matrix *transposed = matrix_transpose(output_layer_weights);
    Matrix *hidden_errors = matrix_mult(transposed, output_error);
    free_matrix(transposed);

    // calculate changes in output layer weights
    Matrix *d_sigmoid_m = matrix_copy(output_layer);

    map(d_sigmoid_m, d_sigmoid);
    Matrix *multiplied1 = entry_mult(output_error, d_sigmoid_m);
    transposed = matrix_transpose(hidden_layer);
    Matrix *multiplied2 = matrix_mult(multiplied1, transposed);
    matrix_scale(multiplied2, learning_rate);
    Matrix *added = matrix_add(output_layer_weights, multiplied2);

    free_matrix(output_layer_weights);
    output_layer_weights = added;

    free_matrix(d_sigmoid_m);
    free_matrix(multiplied1);
    free_matrix(multiplied2);
    free_matrix(transposed);

    // calculate changes in hidden layer weights

    d_sigmoid_m = matrix_copy(hidden_layer);
    map(d_sigmoid_m, d_sigmoid);
    multiplied1 = entry_mult(hidden_errors, d_sigmoid_m);
    transposed = matrix_transpose(input_layer);
    multiplied2 = matrix_mult(multiplied1, transposed);
    matrix_scale(multiplied2, learning_rate);
    added = matrix_add(hidden_layer_weights, multiplied2);

    free_matrix(hidden_layer_weights);
    hidden_layer_weights = added;

    free_matrix(d_sigmoid_m);
    free_matrix(multiplied1);
    free_matrix(multiplied2);
    free_matrix(transposed);

    free_matrix(hidden_errors);
    free_matrix(output_error);
    free_matrix(hidden_layer);
    free_matrix(output_layer);
    free_matrix(softmaxxed);
    free_matrix(input_layer);
    free_matrix(label);
  }

}
