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

    Matrix w0 = matrix_new(3, 2);
    Matrix w1 = matrix_new(1, 3);
    Matrix b0 = matrix_new(3, 1);
    Matrix b1 = matrix_new(1, 1);

    matrix_set(w0, 0, 0,  1);
    matrix_set(w0, 0, 1,  0);
    matrix_set(w0, 1, 0,  2);
    matrix_set(w0, 1, 1,  2);
    matrix_set(w0, 2, 0,  0);
    matrix_set(w0, 2, 1,  1);

    matrix_set(w1, 0, 0,  4);
    matrix_set(w1, 0, 1,  1);
    matrix_set(w1, 0, 2,  3);

    matrix_set(b0, 0, 0,  2);
    matrix_set(b0, 1, 0,  3);
    matrix_set(b0, 2, 0, -5);

    matrix_set(b1, 0, 0, -6);

    set_w(&network, 0, w0);
    set_w(&network, 1, w1);
    set_b(&network, 0, b0);
    set_b(&network, 1, b1);


    Matrix x0 = matrix_new(2, 1);
    matrix_set(x0, 0, 0, 2);
    matrix_set(x0, 1, 0, 3);

    Matrix x1 = matrix_new(2, 1);
    matrix_set(x1, 0, 0, -4);
    matrix_set(x1, 1, 0, -1);

    Matrix x2 = matrix_new(2, 1);
    matrix_set(x2, 0, 0, 6);
    matrix_set(x2, 1, 0, 5);

    Matrix y0, y1, y2;
    y0 = network_feed(network, x0);
    y1 = network_feed(network, x1);
    y2 = network_feed(network, x2);

    printf("--- 0 ---\n");
    matrix_print(x0);
    printf("\n");
    matrix_print(y0);
    printf("\n");

    printf("--- 1 ---\n");
    matrix_print(x1);
    printf("\n");
    matrix_print(y1);
    printf("\n");

    printf("--- 2 ---\n");
    matrix_print(x2);
    printf("\n");
    matrix_print(y2);
    printf("\n");

    matrix_delete(&x0);
    matrix_delete(&x1);
    matrix_delete(&x2);
    matrix_delete(&y0);
    matrix_delete(&y1);
    matrix_delete(&y2);

    network_delete(&network);

    /* MnistSet training_set, test_set; */
    /* mnist_new(&training_set, &test_set); */

    /* mnist_delete(&training_set); */
    /* mnist_delete(&test_set); */

    return 0;
}
