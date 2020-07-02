#ifndef NETWORK_H
#define NETWORK_H
#include "matrix.h"

typedef double (*ScalarFunc)(double);

typedef struct {
    int layers;
    Matrix * weights;
    Matrix * biases;
    ScalarFunc activation_function;
    ScalarFunc activation_prime;
} Network;

Network network_new(
    int layers,
    int * layer_sizes,
    ScalarFunc activation_function,
    ScalarFunc activation_prime
);

Matrix network_feed(
    const Network network,
    const Matrix vector
);

void network_learn(
    Network network,
    const Matrix * mini_batch,
    int mini_batch_size,
    double learing_rate
);

void network_delete(
    Network * p_network
);

#endif
