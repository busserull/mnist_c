#ifndef MATRIX_H
#define MATRIX_H

typedef struct {
    int x;
    int y;
    double * p_data;
} Matrix;

Matrix matrix_new(int dim_x, int dim_y);

void matrix_set(Matrix * p_matrix, int x, int y, double value);

Matrix matrix_dot(Matrix * p_left, Matrix * p_right);

void matrix_print(Matrix * p_matrix);

void matrix_delete(Matrix * p_matrix);

#endif
