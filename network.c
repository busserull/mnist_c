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
    const Matrix correct_output,
    Matrix * delta_nabla_w,
    Matrix * delta_nabla_b
){
    int nabla_size = network.layers - 1;
    Matrix * kernels = (Matrix *)malloc(nabla_size * sizeof(Matrix));
    Matrix * acts = (Matrix *)malloc((nabla_size + 1) * sizeof(Matrix));

    Matrix act = matrix_deep_copy(input);
    acts[0] = act;

    /* Feed forward */
    for(int i = 0; i < nabla_size; i++){
        Matrix kernel = matrix_dot(network.weights[i], act);
        matrix_inplace_add(kernel, network.biases[i]);
        kernels[i] = kernel;

        act = matrix_deep_copy(kernel);
        matrix_inplace_apply(act, network.activation_function);
        acts[i + 1] = act;
    }

    /* Propagate backward */
    Matrix error = matrix_deep_copy(acts[nabla_size]);
    matrix_inplace_sub(error, correct_output); // Cost derivative

    Matrix kernel_prime = matrix_deep_copy(kernels[nabla_size - 1]);
    matrix_inplace_apply(kernel_prime, network.activation_prime);
    matrix_inplace_hadamard(error, kernel_prime);
    matrix_delete(&kernel_prime);

    Matrix a_t = matrix_transpose(acts[nabla_size - 1]);
    delta_nabla_w[nabla_size - 1] = matrix_dot(error, a_t);
    delta_nabla_b[nabla_size - 1] = error;
    matrix_delete(&a_t);

    for(int i = nabla_size - 2; i >= 0; i--){
        kernel_prime = matrix_deep_copy(kernels[i]);
        matrix_inplace_apply(kernel_prime, network.activation_prime);

        Matrix w_t = matrix_transpose(network.weights[i + 1]);
        error = matrix_dot(w_t, error);
        matrix_inplace_hadamard(error, kernel_prime);

        a_t = matrix_transpose(acts[i]);
        delta_nabla_w[i] = matrix_dot(error, a_t);
        delta_nabla_b[i] = error;

        matrix_delete(&kernel_prime);
        matrix_delete(&w_t);
        matrix_delete(&a_t);
    }

    /* Cleanup */
    for(int i = 0; i < nabla_size; i++){
        matrix_delete(kernels + i);
    }
    for(int i = 0; i < nabla_size + 1; i++){
        matrix_delete(acts + i);
    }
    free(kernels);
    free(acts);
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

        matrix_inplace_scramble(network.weights[i]);
        matrix_inplace_scramble(network.biases[i]);
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
    const Matrix * mini_batch_labels,
    int mini_batch_size,
    double learning_rate
){
    int nabla_size = network.layers - 1;
    Matrix * nabla_w = (Matrix *)malloc(nabla_size * sizeof(Matrix));
    Matrix * nabla_b = (Matrix *)malloc(nabla_size * sizeof(Matrix));
    for(int i = 0; i < nabla_size; i++){
        nabla_w[i] = matrix_zero_from(network.weights[i]);
        nabla_b[i] = matrix_zero_from(network.biases[i]);
    }
    Matrix * delta_nabla_w = (Matrix *)malloc(nabla_size * sizeof(Matrix));
    Matrix * delta_nabla_b = (Matrix *)malloc(nabla_size * sizeof(Matrix));

    for(int i = 0; i < mini_batch_size; i++){
        network_backpropagate(
            network,
            mini_batch[i],
            mini_batch_labels[i],
            delta_nabla_w,
            delta_nabla_b
        );
        for(int j = 0; j < network.layers - 1; j++){
            matrix_inplace_add(nabla_w[j], delta_nabla_w[j]);
            matrix_inplace_add(nabla_b[j], delta_nabla_b[j]);
            matrix_delete(delta_nabla_w + j);
            matrix_delete(delta_nabla_b + j);
        }
    }

    for(int i = 0; i < network.layers - 1; i++){
        matrix_inplace_scale(nabla_w[i], learning_rate);
        matrix_inplace_scale(nabla_b[i], learning_rate);
        matrix_inplace_sub(network.weights[i], nabla_w[i]);
        matrix_inplace_sub(network.biases[i], nabla_b[i]);
    }

    /* Cleanup */
    for(int i = 0; i < nabla_size; i++){
        matrix_delete(nabla_w + i);
        matrix_delete(nabla_b + i);
    }
    free(nabla_w);
    free(nabla_b);
    free(delta_nabla_w);
    free(delta_nabla_b);
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
