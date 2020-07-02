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

static void network_backpropagate(
    const Network network,
    const Matrix input,
    Matrix ** p_gradient_weights,
    Matrix ** p_gradient_biases
){
    Matrix * kernels = (Matrix *)malloc((network.layers - 1) * sizeof(Matrix));
    Matrix * acts = (Matrix *)malloc((network.layers - 1) * sizeof(Matrix));

    Matrix prev = matrix_deep_copy(input);
    for(int i = 0; i < network.layers - 1; i++){
        Matrix kernel = matrix_dot(network.weights[i], prev);
        matrix_inplace_add(kernel, network.biases[i]);
        kernels[i] = kernel;

        Matrix act = matrix_deep_copy(kernel);
        matrix_inplace_apply(act, network.activation_function);
        acts[i] = act;

        matrix_delete(&prev);
        prev = act;
    }
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
        matrix_inplace_add(act, network.biases[i]);
        matrix_inplace_apply(act, network.activation_function);
    }

    return act;
}

void network_learn(
    Network network,
    const Matrix * mini_batch,
    int mini_batch_size,
    double learning_rate
){

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
