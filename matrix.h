#ifndef MATRIX_H
#define MATRIX_H

typedef struct {
    int x;
    int y;
    double * p_data;
} Matrix;

Matrix matrix_new(int dim_x, int dim_y);

double matrix_get(const Matrix * p_matrix, int x, int y);

void matrix_set(Matrix * p_matrix, int x, int y, double value);

Matrix matrix_dot(const Matrix * p_left, const Matrix * p_right);

void matrix_print(const Matrix * p_matrix);

void matrix_delete(Matrix * p_matrix);

#endif
