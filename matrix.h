#ifndef MATRIX_H
#define MATRIX_H

typedef struct {
    int x;
    int y;
    double * data;
} Matrix;

Matrix matrix_new(int rows, int columns);

Matrix matrix_deep_copy(const Matrix matrix);

double matrix_get(const Matrix matrix, int x, int y);

void matrix_set(Matrix matrix, int x, int y, double value);

Matrix matrix_dot(const Matrix left, const Matrix right);

void matrix_add_inplace(Matrix left, const Matrix right);

void matrix_print(const Matrix matrix);

void matrix_delete(Matrix * p_matrix);

#endif
