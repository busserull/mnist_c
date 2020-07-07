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
    Matrix ** p_nabla_weights,
    Matrix ** p_nabla_biases
){
    Matrix * kernels = (Matrix *)malloc((network.layers - 1) * sizeof(Matrix));
    Matrix * acts = (Matrix *)malloc(network.layers * sizeof(Matrix));

    acts[0] = matrix_deep_copy(input);
    Matrix prev = matrix_deep_copy(input);

    /* Feed forward */
    for(int i = 0; i < network.layers - 1; i++){
        Matrix kernel = matrix_dot(network.weights[i], prev);
        matrix_inplace_add(kernel, network.biases[i]);
        kernels[i] = kernel;

        Matrix act = matrix_deep_copy(kernel);
        matrix_inplace_apply(act, network.activation_function);
        acts[i + 1] = act;

        matrix_delete(&prev);
        prev = act;
    }

    /* Propagate backward */
    Matrix kernel_prime = matrix_deep_copy(kernels[network.layers - 2]);
    matrix_inplace_apply(kernel_prime, network.activation_prime);
    Matrix error = matrix_deep_copy(acts[network.layers - 1]);
    matrix_inplace_sub(error, correct_output);
    matrix_inplace_hadamard(error, kernel_prime);

    Matrix act_t = matrix_transpose(acts[network.layers - 2]);
    *p_nabla_biases[network.layers - 1] = error;
    *p_nabla_weights[network.layers - 1] = matrix_dot(error, act_t);

    matrix_delete(&kernel_prime);
    matrix_delete(&act_t);

    for(int i = network.layers - 2; i >= 0; i--){
        kernel_prime = matrix_deep_copy(kernels[i]);
        matrix_inplace_apply(kernel_prime, network.activation_prime);
        Matrix weight_t = matrix_transpose(network.weights[i + 1]);
        Matrix error_prop = matrix_dot(weight_t, error);
        matrix_inplace_hadamard(error_prop, kernel_prime);

        act_t = matrix_transpose(acts[i - 1]);
        *p_nabla_biases[i] = error_prop;
        *p_nabla_weights[i] = matrix_dot(error_prop, act_t);

        matrix_delete(&kernel_prime);
        matrix_delete(&weight_t);
        matrix_delete(&act_t);
    }

    /* Cleanup */
    for(int i = 0; i < network.layers - 2; i++){
        matrix_delete(kernels + i);
    }
    for(int i = 0; i < network.layers - 1; i++){
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
    Matrix * nabla_weights = (Matrix *)malloc(network.layers * sizeof(Matrix));
    Matrix * nabla_biases = (Matrix *)malloc(network.layers * sizeof(Matrix));
    Matrix * delta_nabla_w = (Matrix *)malloc(network.layers * sizeof(Matrix));
    Matrix * delta_nabla_b = (Matrix *)malloc(network.layers * sizeof(Matrix));

    for(int i = 0; i < mini_batch_size; i++){
        network_backpropagate(
            network,
            mini_batch[i],
            mini_batch_labels[i],
            &delta_nabla_w,
            &delta_nabla_b
        );
        for(int j = 0; j < network.layers - 1; j++){
            matrix_inplace_add(nabla_weights[j], delta_nabla_w[j]);
            matrix_inplace_add(nabla_biases[j], delta_nabla_b[j]);
        }
    }

    for(int i = 0; i < network.layers - 1; i++){
        matrix_inplace_scale(nabla_weights[i], learning_rate);
        matrix_inplace_scale(nabla_biases[i], learning_rate);
        matrix_inplace_sub(network.weights[i], nabla_weights[i]);
        matrix_inplace_sub(network.biases[i], nabla_biases[i]);
    }

    free(nabla_weights);
    free(nabla_biases);
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
