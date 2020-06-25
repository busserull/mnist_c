#ifndef MATRIX_H
#define MATRIX_H

typedef struct {
    int x;
    int y;
    double * p_data;
} Matrix;

Matrix matrix_new(int dim_x, int dim_y);

Matrix matrix_deep_copy(const Matrix * p_matrix);

double matrix_get(const Matrix * p_matrix, int x, int y);

void matrix_set(Matrix * p_matrix, int x, int y, double value);

void matrix_add(Matrix * p_left, const Matrix * p_right);

Matrix matrix_dot(const Matrix * p_left, const Matrix * p_right);

void matrix_print(const Matrix * p_matrix);

void matrix_delete(Matrix * p_matrix);

#endif
