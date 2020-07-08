#include "matrix.h"
#include "mnist.h"
#include "network.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void set_w(Network *, int, Matrix);
void set_b(Network *, int, Matrix);

double sigmoid(double v){
    return 1.0 / (1.0 + exp(-v));
}

double sigmoid_prime(double v){
    double raised = exp(-v);
    return raised / ((1 + raised) * (1 + raised));
}

void evaluate(const Network network, const MnistSet set){
    /* double correct = 0; */
    /* double total = 0; */

    for(int i = 0; i < set.size; i++){
        Matrix image = mnist_vectorize_image(set.images[i]);
        Matrix guess = network_feed(network, image);

        mnist_print_image(set.images[i]);
        printf("Correct: %d, Guess: ", set.labels[i]);

        printf("\n");
        matrix_print(guess);
        /* int max_index = 0; */
        /* double max_guess = matrix_get(guess, 0, 0); */
        /* for(int y = 0; y < guess.y; y++){ */
        /*     if(matrix_get(guess, 0, y) > max_guess){ */
        /*         max_guess = matrix_get(guess, 0, y); */
        /*         max_index = y; */
        /*     } */
        /* } */
        /* printf("%d (%.2f)\n", max_index, max_guess); */

        matrix_delete(&image);
        matrix_delete(&guess);
    }
}

int main(){
    srand(0);

    MnistSet training_set, test_set;
    mnist_new(&training_set, &test_set);
    int layers[] = {784, 15, 15, 10};
    Network network = network_new(4, layers, sigmoid, sigmoid_prime);

    int old_size = test_set.size;
    printf("--- Before training ---\n");
    test_set.size = 5;
    evaluate(network, test_set);
    test_set.size = old_size;

    for(int epoch = 0; epoch < 10; epoch++){
        Matrix * images = (Matrix *)malloc(100 * sizeof(Matrix));
        Matrix * labels = (Matrix *)malloc(100 * sizeof(Matrix));
        for(int i = 0; i < 100; i++){
            images[i] = mnist_vectorize_image(training_set.images[i]);
            labels[i] = mnist_vectorize_label(training_set.labels[i]);
        }

        network_learn(network, images, labels, 100, 0.3);

        for(int i = 0; i < 100; i++){
            matrix_delete(images + i);
            matrix_delete(labels + i);
        }
        free(images);
        free(labels);
    }

    printf("--- After training ---\n");
    test_set.size = 5;
    evaluate(network, test_set);
    test_set.size = old_size;

    network_delete(&network);
    mnist_delete(&training_set);
    mnist_delete(&test_set);

    return 0;
}
