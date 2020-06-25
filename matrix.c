#include "matrix.h"
#include <stdlib.h>
#include <stdio.h>

#ifdef DEBUG
#include <assert.h>
#endif

Matrix matrix_new(int dim_x, int dim_y){
    Matrix matrix;
    matrix.x = dim_x;
    matrix.y = dim_y;
    matrix.p_data = (double *)malloc(dim_x * dim_y * sizeof(double));
    return matrix;
}

Matrix matrix_deep_copy(const Matrix * p_matrix){
    Matrix matrix = matrix_new(p_matrix->x, p_matrix->y);
    for(int i = 0; i < matrix.x * matrix.y; i++){
        matrix.p_data[i] = p_matrix->p_data[i];
    }
    return matrix;
}

double matrix_get(const Matrix * p_matrix, int x, int y){
    return p_matrix->p_data[x * p_matrix->y + y];
}

void matrix_set(Matrix * p_matrix, int x, int y, double value){
#ifdef DEBUG
    assert(x >= 0 && x < p_matrix->x);
    assert(y >= 0 && y < p_matrix->y);
#endif
    int index = x * p_matrix->y + y;
    p_matrix->p_data[index] = value;
}

void matrix_add(Matrix * p_left, const Matrix * p_right){
#ifdef DEBUG
    assert(p_left->x == p_right->x);
    assert(p_left->y == p_right->y);
#endif
    for(int i = 0; i < p_left->x * p_left->y; i++){
        p_left->p_data[i] += p_right->p_data[i];
    }
}

Matrix matrix_dot(const Matrix * p_left, const Matrix * p_right){
#ifdef DEBUG
    assert(p_left->y == p_right->x);
#endif
    Matrix product = matrix_new(p_left->x, p_right->y);
    for(int x = 0; x < product.x; x++){
        for(int y = 0; y < product.y; y++){
            product.p_data[x * product.y + y] = 0.0;

            for(int i = 0; i < p_left->y; i++){
                double v_left = p_left->p_data[x * p_left->y + i];
                double v_right = p_right->p_data[i * p_right->y + y];
                product.p_data[x * product.y + y] += v_left * v_right;
            }
        }
    }

    return product;
}

void matrix_print(const Matrix * p_matrix){
    for(int x = 0; x < p_matrix->x; x++){
        for(int y = 0; y < p_matrix->y; y++){
            double value = p_matrix->p_data[x * p_matrix->y + y];
            printf("%8.2f", value);
        }
        printf("\n");
    }
    printf("\n");
}

void matrix_delete(Matrix * p_matrix){
    free(p_matrix->p_data);
    p_matrix->x = 0;
    p_matrix->y = 0;
}
