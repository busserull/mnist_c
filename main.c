#include "matrix.h"
#include "mnist.h"
#include "network.h"
#include <stdio.h>
#include <math.h>

void set_w(Network *, int, Matrix);
void set_b(Network *, int, Matrix);

double sigmoid(double v){
    return 1.0 / (1.0 + exp(-v));
}

int main(){
    int layers[] = {2, 3, 1};
    Network network = network_new(3, layers, sigmoid, NULL);

    Matrix A = matrix_new(3, 2);
    matrix_set(A, 0, 0, 1);
    matrix_set(A, 0, 1, 2);
    matrix_set(A, 1, 0, 3);
    matrix_set(A, 1, 1, 4);
    matrix_set(A, 2, 0, 5);
    matrix_set(A, 2, 1, 6);

    Matrix At = matrix_transpose(A);

    matrix_print(A);
    matrix_print(At);
    matrix_inplace_scale(At, 2.0);
    matrix_print(At);

    matrix_delete(&A);
    matrix_delete(&At);

    network_delete(&network);

    /* MnistSet training_set, test_set; */
    /* mnist_new(&training_set, &test_set); */

    /* mnist_delete(&training_set); */
    /* mnist_delete(&test_set); */

    return 0;
}
