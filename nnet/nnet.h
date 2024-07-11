#include <stdint.h>

#define INPUTSIZE 784 // 28 * 28
#define N_HIDDENLAYERS 1
#define N_NODES 300
#define N_OUTPUT 10
#define TRAINING_SETS 60000
#define N_IMAGES 30000

typedef struct {
  uint8_t image[INPUTSIZE];
  uint8_t label;
} input_data;

void load_training_data(input_data *training_data, FILE *training_images, FILE *training_labels);

double sigmoid(double x);

double d_sigmoid(double x);
