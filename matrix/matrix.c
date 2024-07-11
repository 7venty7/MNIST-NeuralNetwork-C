#include "matrix.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

void matrix_zero(Matrix *m) {
  for (int i = 0; i < m->rows; i++) {
    for (int j = 0; j < m->cols; j++) {
      m->values[i][j] = 0;
    }
  }
}

void matrix_scale(Matrix *m, double scalar) {
  for (int i = 0; i < m->rows; i++) {
    for (int j = 0; j < m->cols; j++) {
      m->values[i][j] = scalar * m->values[i][j];
    }
  }
}

Matrix *matrix_copy(Matrix *m) {
  Matrix *matrix = matrix_init(m->rows, m->cols);

  for (int i = 0; i < m->rows; i++) {
    for (int j = 0; j < m->cols; j++) {
      matrix->values[i][j] = m->values[i][j];
    }
  }

  return matrix;
}

Matrix *matrix_transpose(Matrix *m) {
  Matrix *transposed = matrix_init(m->cols, m->rows);

  for (int i = 0; i < m->rows; i++) {
    for (int j = 0; j < m->cols; j++) {
      transposed->values[j][i] = m->values[i][j];
    }
  }

  return transposed;
}

void map(Matrix *m, double (*func)(double)) {
  for (int i = 0; i < m->rows; i++) {
    for (int j = 0; j < m->cols; j++) {
      m->values[i][j] = func(m->values[i][j]);
    }
  }
}

Matrix *entry_mult(Matrix *m1, Matrix *m2) {
  if (m1->rows != m2->rows || m1->cols != m2->cols) {
    printf("Invalid rows and columns for element-wise multiplication.\n");
    exit(1);
  }

  Matrix *result = matrix_init(m1->rows, m1->cols);

  for (int i = 0; i < m1->rows; i++) {
    for (int j = 0; j < m1->cols; j++) {
      result->values[i][j] = m1->values[i][j] * m2->values[i][j];
    }
  }

  return result;
}

Matrix *matrix_mult(Matrix *m1, Matrix *m2) {
  if (m1->cols != m2->rows) {
    printf("Invalid rows and columns for matrix multiplication\n");
    exit(1);
  }

  Matrix *result = matrix_init(m1->rows, m2->cols);
  matrix_zero(result);

  for (int i = 0; i < m1->rows; i++) {
    for (int j = 0; j < m2->cols; j++) {
      for (int k = 0; k < m2->rows; k++) {
        result->values[i][j] += m1->values[i][k] * m2->values[k][j];
      }
    }
  }

  return result;
}

Matrix *matrix_add(Matrix *m1, Matrix *m2) {
  if (m1->rows != m2->rows || m1->cols != m2->cols) {
    printf("Invalid rows and columns for matrix addition\n");
    exit(1);
  }

  Matrix *result = matrix_init(m1->rows, m1->cols);

  for (int i = 0; i < m1->rows; i++) {
    for (int j = 0; j < m1->cols; j++) {
      result->values[i][j] = m1->values[i][j] + m2->values[i][j];
    }
  }

  return result;
}

Matrix *matrix_subtract(Matrix *m1, Matrix *m2) {
  if (m1->rows != m2->rows || m1->cols != m2->cols) {
    printf("Invalid rows and columns for matrix subtraction\n");
    exit(1);
  }

  Matrix *result = matrix_init(m1->rows, m1->cols);

  for (int i = 0; i < m1->rows; i++) {
    for (int j = 0; j < m1->cols; j++) {
      result->values[i][j] = m1->values[i][j] - m2->values[i][j];
    }
  }

  return result;
}

void randomise_matrix(Matrix *m) {
  for (int i = 0; i < m->rows; i++) {
    for (int j = 0; j < m->cols; j++) {
      m->values[i][j] = init_weight();
    }
  }
}

Matrix *matrix_init(int rows, int cols) {
  Matrix *matrix = malloc(sizeof(Matrix));
  matrix->rows = rows;
  matrix->cols = cols;

  matrix->values = malloc(rows * sizeof(double *));
  
  for (int i = 0; i < rows; i++) {
    matrix->values[i] = malloc(cols * sizeof(double));
  }

  return matrix;
}

void free_matrix(Matrix *matrix) {
  for (int i = 0; i < matrix->rows; i++) {
    free(matrix->values[i]);
  }

  free(matrix->values);
  free(matrix);

  matrix = NULL;
}

double init_weight() {
  double weight = (rand() - (RAND_MAX / 2)) / (RAND_MAX / 2);

  return weight;
}

int max(Matrix *m) {
  if (m->cols != 1) {
    printf("Invalid matrix for max\n");
  }

  int max = 0;

  for (int i = 0; i < m->rows; i++) {
    if (m->values[i][0] > m->values[max][0]) {
      max = i;
    }
  }

  return max;
}

Matrix *softmax(Matrix *m) {
  double sigma = 0;

  for (int i = 0; i < m->rows; i++) {
    for (int j = 0; j < m->cols; j++) {
      sigma += exp(m->values[i][j]);
    }
  }

  Matrix *result = matrix_init(m->rows, m->cols);

  for (int i = 0; i < m->rows; i++) {
    for (int j = 0; j < m->cols; j++) {
      result->values[i][j] = exp(m->values[i][j]) / sigma;
    }
  }

  return result;
}

double dist(double min, double max) {
  double difference = max - min;
  int scale = 10000;
  int scaled_difference = scale * difference;
  return min + (1.0 * (rand() % scaled_difference) / scale);
}

void randomise2(Matrix *m, int n) {
  double min = -1 / sqrt(n);
  double max = 1 / sqrt(n);

  for (int i = 0; i < m->rows; i++) {
    for (int j = 0; j < m->cols; j++) {
      m->values[i][j] = dist(min, max);
    }
  }
}
