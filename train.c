#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include "nnet/nnet.h"

int main(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    srand(tv.tv_usec * tv.tv_sec);

    const double learning_rate = 0.2f;

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

    Matrix *output_layer_bias = matrix_init(N_OUTPUT, 1);
    Matrix *hidden_layer_bias = matrix_init(N_NODES, 1);

    randomise2(hidden_layer_weights, N_NODES);
    randomise2(output_layer_weights, N_OUTPUT);

    randomise2(output_layer_bias, N_OUTPUT);
    randomise2(hidden_layer_bias, N_NODES);

    for (int i = 0; i < N_IMAGES; i++) {
        // initial forward pass
        Matrix *input_layer = matrix_init(INPUTSIZE, 1);
        load_training_data(&training_data, training_images, training_labels);

        for (int j = 0; j < INPUTSIZE; j++) {
            input_layer->values[j][0] = (double) training_data.image[j] / 256;
        }

        Matrix *hidden_layer_in = matrix_mult(hidden_layer_weights, input_layer);
        Matrix *hidden_layer = matrix_add(hidden_layer_in, hidden_layer_bias);
        map(hidden_layer, sigmoid);

        Matrix *output_layer_in = matrix_mult(output_layer_weights, hidden_layer);
        Matrix *output_layer = matrix_add(output_layer_in, output_layer_bias);
        map(output_layer, sigmoid);

        Matrix *softmaxxed = softmax(output_layer);
        int prediction = max(softmaxxed);
        printf("Label: %d, Prediction: %d\n", training_data.label, prediction);

        // create the label vector
        Matrix *label = matrix_init(N_OUTPUT, 1);
        label->values[training_data.label][0] = 1;

        // backprop

        // compute errors

        Matrix *output_error = matrix_subtract(label, output_layer);

        Matrix *transposed = matrix_transpose(output_layer_weights);
        Matrix *hidden_errors = matrix_mult(transposed, output_error);
        free_matrix(transposed);

        // calculate changes in output layer weights and biases

        Matrix *d_sigmoid_m = matrix_copy(output_layer);

        map(d_sigmoid_m, d_sigmoid);
        Matrix *multiplied = entry_mult(output_error, d_sigmoid_m);
        transposed = matrix_transpose(hidden_layer);
        Matrix *delta_weights = matrix_mult(multiplied, transposed);
        matrix_scale(delta_weights, learning_rate);
        Matrix *adjusted_weights = matrix_add(output_layer_weights, delta_weights);

        Matrix *delta_bias = matrix_copy(multiplied);
        matrix_scale(delta_bias, learning_rate);
        Matrix *adjusted_bias = matrix_add(output_layer_bias, delta_bias);

        free_matrix(output_layer_weights);
        free_matrix(output_layer_bias);
        output_layer_weights = adjusted_weights;
        output_layer_bias = adjusted_bias;

        free_matrix(d_sigmoid_m);
        free_matrix(multiplied);
        free_matrix(delta_weights);
        free_matrix(transposed);
        free_matrix(delta_bias);

        // calculate changes in hidden layer weights and biases

        d_sigmoid_m = matrix_copy(hidden_layer);
        map(d_sigmoid_m, d_sigmoid);
        multiplied = entry_mult(hidden_errors, d_sigmoid_m);
        transposed = matrix_transpose(input_layer);
        delta_weights = matrix_mult(multiplied, transposed);
        matrix_scale(delta_weights, learning_rate);
        adjusted_weights = matrix_add(hidden_layer_weights, delta_weights);

        delta_bias = matrix_copy(multiplied);
        matrix_scale(delta_bias, learning_rate);
        adjusted_bias = matrix_add(hidden_layer_bias, delta_bias);

        free_matrix(hidden_layer_weights);
        free_matrix(hidden_layer_bias);
        hidden_layer_weights = adjusted_weights;
        hidden_layer_bias = adjusted_bias;

        free_matrix(d_sigmoid_m);
        free_matrix(multiplied);
        free_matrix(delta_weights);
        free_matrix(transposed);
        free_matrix(delta_bias);

        free_matrix(hidden_errors);
        free_matrix(output_error);
        free_matrix(hidden_layer_in);
        free_matrix(hidden_layer);
        free_matrix(output_layer_in);
        free_matrix(output_layer);
        free_matrix(softmaxxed);
        free_matrix(input_layer);
        free_matrix(label);
    }

    // perform testing

    FILE *testing_images;
    FILE *testing_labels;

    testing_images = fopen("t10k-images-idx3-ubyte", "rb");
    testing_labels = fopen("t10k-labels-idx1-ubyte", "rb");

    if (testing_images == NULL || testing_labels == NULL) {
        printf("Error loading files\n");
        exit(1);
    }

    fseek(testing_images, 16, SEEK_SET);
    fseek(testing_labels, 8, SEEK_SET);

    input_data testing_data;

    int correct = 0;

    for (int i = 0; i < N_TESTING; i++) {
        load_training_data(&testing_data, testing_images, testing_labels);
        int prediction = predict(testing_data, hidden_layer_weights, hidden_layer_bias, output_layer_weights, output_layer_bias);

        if (prediction == testing_data.label) {
            correct++;
        }

        printf("Label: %d, Prediction: %d\n", testing_data.label, prediction);
    }

    printf("Accuracy of model: %lf\n", (double) correct / N_TESTING);

    FILE *hidden_weights_out;
    FILE *hidden_bias_out;
    FILE *output_weights_out;
    FILE *output_bias_out;

    hidden_weights_out = fopen("hidden_layer_weights.csv", "w");
    hidden_bias_out = fopen("hidden_layer_bias.csv", "w");
    output_weights_out = fopen("output_layer_weights.csv", "w");
    output_bias_out = fopen("output_layer_bias.csv", "w");

    if (hidden_weights_out == NULL || hidden_bias_out == NULL || output_weights_out == NULL || output_bias_out == NULL) {
        printf("Error creating output files\n");
        exit(1);
    }

    for (int i = 0; i < N_NODES; i++) {
        for (int j = 0; j < INPUTSIZE; j++) {
            fprintf(hidden_weights_out, "%lf", hidden_layer_weights->values[i][j]);

            if (j != INPUTSIZE - 1) {
                fprintf(hidden_weights_out, ",");
            }
        }

        if (i != N_NODES - 1) {
            fprintf(hidden_weights_out, "\n");
        }
    }

    for (int i = 0; i < N_NODES; i++) {
        fprintf(hidden_bias_out, "%lf", hidden_layer_bias->values[i][0]);

        if (i != N_NODES - 1) {
            fprintf(hidden_bias_out, "\n");
        }
    }

    for (int i = 0; i < N_OUTPUT; i++) {
        for (int j = 0; j < N_NODES; j++) {
            fprintf(output_weights_out, "%lf", output_layer_weights->values[i][j]);

            if (j != N_NODES - 1) {
                fprintf(output_weights_out, ",");
            }
        }

        if (i != N_OUTPUT - 1) {
            fprintf(output_weights_out, "\n");
        }
    }

    for (int i = 0; i < N_OUTPUT; i++) {
        fprintf(output_bias_out, "%lf", output_layer_bias->values[i][0]);

        if (i != N_OUTPUT - 1) {
            fprintf(output_bias_out, "\n");
        }
    }

    fclose(training_images);
    fclose(training_labels);
    fclose(testing_images);
    fclose(testing_labels);
    fclose(hidden_weights_out);
    fclose(hidden_bias_out);
    fclose(output_weights_out);
    fclose(output_bias_out);

    free_matrix(hidden_layer_weights);
    free_matrix(hidden_layer_bias);
    free_matrix(output_layer_weights);
    free_matrix(output_layer_bias);
}
