#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "nnet.h"

void load_training_data(input_data *training_data, FILE *training_images, FILE *training_labels) {
  fread(training_data->image, sizeof(uint8_t), INPUTSIZE, training_images);
  fread(&training_data->label, sizeof(uint8_t), 1, training_labels);
}

double sigmoid(double x) {
  return (1 / (1 + exp(-1 * x)));
}

double d_sigmoid(double x) {
  return (x * (1 - x));
}
