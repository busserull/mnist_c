#include "matrix.h"
#include "mnist.h"
#include <stdio.h>

int main(){
    MNISTData mnist = mnist_new();

    printf("Training size: %d\n", mnist.size_training);
    for(int i = 0; i < 20; i++){
        printf("%d ", mnist.training_labels[i]);
    }
    printf("\n");

    printf("Test size: %d\n", mnist.size_test);
    for(int i = 0; i < 20; i++){
        printf("%d ", mnist.test_labels[i]);
    }
    printf("\n\n");

    for(int i = 0; i < 5; i++){
        mnist_print_image(mnist.training_images + i);
    }

    mnist_delete(&mnist);

    return 0;
}
