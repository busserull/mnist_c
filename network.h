#ifndef NETWORK_H
#define NETWORK_H
#include "matrix.h"

typedef double (*ScalarFunc)(double);

typedef struct {
    int layers;
    Matrix * weights;
    Matrix * biases;
    ScalarFunc activation;
    ScalarFunc activation_prime;
} Network;

Network network_new(int layers, int * layer_sizes);

Matrix network_feed(const Network * p_network, const Matrix * p_vector);

void network_delete(Network * p_network);

#endif
