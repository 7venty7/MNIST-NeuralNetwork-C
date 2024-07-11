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

int predict(input_data input, Matrix *hidden_weights, Matrix *hidden_bias, Matrix *output_weights, Matrix *output_bias) {
    Matrix *input_layer = matrix_init(INPUTSIZE, 1);

    for (int j = 0; j < INPUTSIZE; j++) {
        input_layer->values[j][0] = (double) input.image[j] / 256;
    }

    Matrix *hidden_layer_in = matrix_mult(hidden_weights, input_layer);
    Matrix *hidden_layer = matrix_add(hidden_layer_in, hidden_bias);
    map(hidden_layer, sigmoid);

    Matrix *output_layer_in = matrix_mult(output_weights, hidden_layer);
    Matrix *output_layer = matrix_add(output_layer_in, output_bias);
    map(output_layer, sigmoid);

    Matrix *softmaxxed = softmax(output_layer);
    int prediction = max(softmaxxed);

    return prediction;
}
