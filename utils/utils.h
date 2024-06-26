#include <stdint.h>

#define INPUTSIZE 28 * 28
#define N_HIDDENLAYERS 2
#define N_NODES 16
#define N_OUTPUT 10
#define TRAINING_SETS 5000

typedef struct {
  uint8_t image[INPUTSIZE];
  uint8_t label;
} data;

/*
 * load_training_data
 * Load the training data from the MNIST files
 */
void load_training_data(data *training_data);

/*
 * matrix_vector_mult
 * An unoptimised matrix vector multiplication algorithm
 * used to apply the weights to the activations of the previous layer
 * for the next layer
 */
void matrix_vector_mult(double **weights, double *activations, double *nxt_layer, int rows, int cols);

/*
 * vec_add
 * Performs vector addition on two arrays
 * used to add in the biases to the activations
 */
void vec_add(double *activations, double *biases, int len);
