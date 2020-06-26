#include "network.h"
#include <stdlib.h>

void set_w(Network * p_network, int layer, Matrix w){
    matrix_delete(p_network->weights + layer);
    p_network->weights[layer] = w;
}

void set_b(Network * p_network, int layer, Matrix b){
    matrix_delete(p_network->biases + layer);
    p_network->biases[layer] = b;
}

Network network_new(
    int layers,
    int * layer_sizes,
    ScalarFunc activation_function,
    ScalarFunc activation_prime
){
    Network network;
    network.layers = layers;
    network.weights = (Matrix *)malloc((layers - 1) * sizeof(Matrix));
    network.biases = (Matrix *)malloc((layers - 1) * sizeof(Matrix));
    network.activation_function = activation_function;
    network.activation_prime = activation_prime;

    for(int i = 0; i < layers - 1; i++){
        int from = layer_sizes[i];
        int to = layer_sizes[i + 1];
        network.weights[i] = matrix_new(to, from);
        network.biases[i] = matrix_new(to, 1);
    }

    return network;
}

Matrix network_feed(const Network network, const Matrix vector){
    Matrix act = matrix_deep_copy(vector);

    for(int i = 0; i < network.layers - 1; i++){
        Matrix next = matrix_dot(network.weights[i], act);
        matrix_delete(&act);
        act = next;
        matrix_add_inplace(act, network.biases[i]);
        // Apply activation function
    }

    return act;
}

void network_delete(Network * p_network){
    for(int i = 0; i < p_network->layers - 1; i++){
        matrix_delete(p_network->weights + i);
        matrix_delete(p_network->biases + i);
    }
    free(p_network->weights);
    free(p_network->biases);
    p_network->layers = 0;
}
