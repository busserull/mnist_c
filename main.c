#include "matrix.h"
#include "mnist.h"
#include "network.h"
#include "random.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define ACTIVATION_ARGUMENT_LIMIT 20

double sigmoid(double v){
    if(v > ACTIVATION_ARGUMENT_LIMIT){
        v = ACTIVATION_ARGUMENT_LIMIT;
    }
    else if(v < -ACTIVATION_ARGUMENT_LIMIT){
        v = -ACTIVATION_ARGUMENT_LIMIT;
    }
    return 1.0 / (1.0 + exp(-v));
}

double sigmoid_prime(double v){
    if(v > ACTIVATION_ARGUMENT_LIMIT){
        v = ACTIVATION_ARGUMENT_LIMIT;
    }
    else if(v < -ACTIVATION_ARGUMENT_LIMIT){
        v = -ACTIVATION_ARGUMENT_LIMIT;
    }
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

void print_progress(int total_size, int remaining_size){
    int percent = ((total_size - remaining_size) * 100) / total_size;
    int progress_bars = (percent * 25) / 100;

    printf("\r  [");
    for(int i = 0; i < progress_bars; i++){
        printf("=");
    }
    for(int i = 0; i < 25 - progress_bars; i++){
        printf(" ");
    }
    printf("] %3d%% ", percent);

    fflush(stdout);

    if(percent == 100){
        printf("\n");
    }
}

#define LAYERS {784, 300, 150, 50, 15, 10}
#define EPOCHS 50
#define MINI_BATCH_SIZE 100
#define LEARNING_RATE 3.0

int main(){
    random_seed(time(NULL));

    MnistSet training_set, test_set;
    mnist_new(&training_set, &test_set);

    int layers[] = LAYERS;
    int n_layers = sizeof(layers) / sizeof(*layers);
    Network network = network_new(n_layers, layers, sigmoid, sigmoid_prime);

    for(int epoch = 0; epoch < EPOCHS; epoch++){
        printf("Epoch %d/%d\n", epoch + 1, EPOCHS);
        fflush(stdout);

        mnist_shuffle(training_set);

        int remaining_items = training_set.size;
        while(remaining_items != 0){
            int mb_size = MINI_BATCH_SIZE;
            if(remaining_items < MINI_BATCH_SIZE){
                mb_size = remaining_items;
            }
            remaining_items -= mb_size;

            Matrix * images = (Matrix *)calloc(mb_size, sizeof(Matrix));
            Matrix * labels = (Matrix *)calloc(mb_size, sizeof(Matrix));
            for(int i = 0; i < mb_size; i++){
                int j = MINI_BATCH_SIZE * epoch + i;
                images[i] = mnist_vectorize_image(training_set.images[j]);
                labels[i] = mnist_vectorize_label(training_set.labels[j]);
            }

            network_learn(network, images, labels, mb_size, LEARNING_RATE);

            print_progress(training_set.size, remaining_items);

            for(int i = 0; i < mb_size; i++){
                matrix_delete(images + i);
                matrix_delete(labels + i);
            }
            free(images);
            free(labels);
        }
    }

    int real_size = training_set.size;
    training_set.size = 10;
    evaluate(network, training_set);
    training_set.size = real_size;

    network_delete(&network);
    mnist_delete(&training_set);
    mnist_delete(&test_set);

    return 0;
}
