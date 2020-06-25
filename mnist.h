#ifndef MNIST_H
#define MNIST_H
#include "matrix.h"

#define MNIST_TRAINING_IMAGE_FILE "mnist/train-images.idx3-ubyte"
#define MNIST_TRAINING_LABEL_FILE "mnist/train-labels.idx1-ubyte"
#define MNIST_TEST_IMAGE_FILE "mnist/t10k-images.idx3-ubyte"
#define MNIST_TEST_LABEL_FILE "mnist/t10k-labels.idx1-ubyte"

typedef struct {
    int size_training;
    int size_test;
    int * training_labels;
    int * test_labels;
    Matrix * training_images;
    Matrix * test_images;
} MNISTData;

MNISTData mnist_new();

void mnist_delete(MNISTData * p_data);

#endif
