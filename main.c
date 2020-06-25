#include "matrix.h"

int main(){
    Matrix a = matrix_new(3, 2);
    matrix_set(&a, 0, 0, 3);
    matrix_set(&a, 0, 1, 2);
    matrix_set(&a, 1, 0, 1);
    matrix_set(&a, 1, 1, -5);
    matrix_set(&a, 2, 0, -2);
    matrix_set(&a, 2, 1, 1);

    Matrix b = matrix_new(2, 2);
    matrix_set(&b, 0, 0, 2);
    matrix_set(&b, 0, 1, -4);
    matrix_set(&b, 1, 0, -3);
    matrix_set(&b, 1, 1, 6);

    matrix_print(&a);
    matrix_print(&b);

    Matrix c = matrix_dot(&a, &b);

    matrix_print(&c);

    matrix_delete(&a);
    matrix_delete(&b);
    matrix_delete(&c);

    return 0;
}
