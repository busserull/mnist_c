#ifndef MATRIX_H
#define MATRIX_H

typedef struct {
    int x;
    int y;
    double * data;
} Matrix;

Matrix matrix_new(int rows, int columns);

Matrix matrix_deep_copy(const Matrix matrix);

Matrix matrix_zero_from(const Matrix matrix);

double matrix_get(const Matrix matrix, int x, int y);

void matrix_set(Matrix matrix, int x, int y, double value);

Matrix matrix_dot(const Matrix left, const Matrix right);

Matrix matrix_transpose(Matrix matrix);

void matrix_inplace_argmax(Matrix matrix);

void matrix_inplace_scramble(Matrix matrix);

void matrix_inplace_add(Matrix left, const Matrix right);

void matrix_inplace_sub(Matrix left, const Matrix right);

void matrix_inplace_scale(Matrix matrix, double factor);

void matrix_inplace_hadamard(Matrix left, const Matrix right);

void matrix_inplace_apply(Matrix matrix, double (*function)(double));

void matrix_print(const Matrix matrix);

void matrix_delete(Matrix * p_matrix);

#endif
