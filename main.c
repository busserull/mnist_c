#include "matrix.h"
#include "mnist.h"
#include <stdio.h>

int main(){
    MnistSet training_set, test_set;
    mnist_new(&training_set, &test_set);

    printf("Training size: %d\n", training_set.size);
    for(int i = 0; i < 20; i++){
        printf("%d ", training_set.labels[i]);
    }
    printf("\n");

    printf("Test size: %d\n", test_set.size);
    for(int i = 0; i < 20; i++){
        printf("%d ", test_set.labels[i]);
    }
    printf("\n");

    for(int i = 0; i < 5; i++){
        mnist_print_image(training_set.images + i);
    }

    mnist_delete(&training_set);
    mnist_delete(&test_set);

    return 0;
}
