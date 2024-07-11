#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../matrix/matrix.h"

#define INPUTSIZE 784 // 28 * 28
#define N_HIDDENLAYERS 1
#define N_NODES 300
#define N_OUTPUT 10
#define N_IMAGES 12000
#define N_TESTING 10000

typedef struct {
    uint8_t image[INPUTSIZE];
    uint8_t label;
} input_data;

void load_training_data(input_data *training_data, FILE *training_images, FILE *training_labels);

double sigmoid(double x);

double d_sigmoid(double x);

int predict(input_data input, Matrix *hidden_weights, Matrix *hidden_bias, Matrix *output_weights, Matrix *output_bias);
