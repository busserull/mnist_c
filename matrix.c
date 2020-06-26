#include "matrix.h"
#include <stdlib.h>
#include <stdio.h>

#ifdef DEBUG
#include <assert.h>
#endif

Matrix matrix_new(int rows, int columns){
    Matrix matrix;
    matrix.x = rows;
    matrix.y = columns;
    matrix.data = (double *)malloc(rows * columns * sizeof(double));
    return matrix;
}

Matrix matrix_deep_copy(const Matrix matrix){
    Matrix copy = matrix_new(matrix.x, matrix.y);
    for(int i = 0; i < matrix.x * matrix.y; i++){
        copy.data[i] = matrix.data[i];
    }
    return copy;
}

double matrix_get(const Matrix matrix, int x, int y){
#ifdef DEBUG
    assert(x >= 0 && x < matrix.x);
    assert(y >= 0 && y < matrix.y);
#endif
    return matrix.data[x * matrix.y + y];
}

void matrix_set(Matrix matrix, int x, int y, double value){
#ifdef DEBUG
    assert(x >= 0 && x < matrix.x);
    assert(y >= 0 && y < matrix.y);
#endif
    matrix.data[x * matrix.y + y] = value;
}

Matrix matrix_dot(const Matrix left, const Matrix right){
#ifdef DEBUG
    assert(left.y == right.x);
#endif
    Matrix product = matrix_new(left.x, right.y);
    for(int x = 0; x < product.x; x++){
        for(int y = 0; y < product.y; y++){
            product.data[x * product.y + y] = 0.0;
            for(int i = 0; i < left.y; i++){
                double v_left = left.data[x * left.y + i];
                double v_right = right.data[i * right.y + y];
                product.data[x * product.y + y] += v_left * v_right;
            }
        }
    }
    return product;
}

void matrix_add_inplace(Matrix left, const Matrix right){
#ifdef DEBUG
    assert(left.x == right.x);
    assert(left.y == right.y);
#endif
    for(int i = 0; i < left.x * left.y; i++){
        left.data[i] += right.data[i];
    }
}

void matrix_print(const Matrix matrix){
    for(int x = 0; x < matrix.x; x++){
        for(int y = 0; y < matrix.y; y++){
            double value = matrix.data[x * matrix.y + y];
            printf("%8.2f", value);
        }
        printf("\n");
    }
    printf("\n");
}

void matrix_delete(Matrix * p_matrix){
    free(p_matrix->data);
    p_matrix->x = 0;
    p_matrix->y = 0;
}
