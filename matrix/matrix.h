typedef struct {
  int rows;
  int cols;
  double **values;
} Matrix;

Matrix *matrix_transpose(Matrix *m);

void map(Matrix *m, double (*func)(double));

void matrix_zero(Matrix *m);

Matrix *entry_mult(Matrix *m1, Matrix *m2);

void matrix_scale(Matrix *m, double scalar);

/*
 * matrix_mult
 * Performs matrix multiplication on 2 input
 * matrices, returning a pointer to the
 * resultant matrix.
 *
 * Matrix *m1       ::the first input matrix
 * Matrix *m2       ::the second input matrix
 */
Matrix *matrix_mult(Matrix *m1, Matrix *m2);

/*
 * matrix_add
 * Performs entry-wise addition on 2 input
 * matrices, returning a pointer to the
 * resultant matrix.
 *
 * Matrix *m1       ::the first input matrix
 * Matrix *m2       ::the second input matrix
 */
Matrix *matrix_add(Matrix *m1, Matrix *m2);

Matrix *matrix_subtract(Matrix *m1, Matrix *m2);

Matrix *matrix_copy(Matrix *m);

/*
 * randomise_matrix
 * Randomises the values of the entries of the
 * input matrix to a value between -1 and 1.
 *
 * Matrix *m        ::the input matrix
 */
void randomise_matrix(Matrix *m);

/*
 * matrix_init
 * Constructor for matrix
 *
 * int rows       ::no. of rows
 * int cols       ::no. of cols
 */
Matrix *matrix_init(int rows, int cols);

/*
 * free_matrix
 * Frees all of the allocated memory
 * for a matrix
 *
 * Matrix *matrix     ::the matrix to free
 */
void free_matrix(Matrix *matrix);

double init_weight();

Matrix *softmax(Matrix *m);

int max(Matrix *m);

double dist(double min, double max);

void randomise2(Matrix *m, int n);
