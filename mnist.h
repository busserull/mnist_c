#ifndef MNIST_H
#define MNIST_H
#include "matrix.h"
#include <stdint.h>

#define MNIST_TRAINING_IMAGE_FILE "mnist/train-images.idx3-ubyte"
#define MNIST_TRAINING_LABEL_FILE "mnist/train-labels.idx1-ubyte"
#define MNIST_TEST_IMAGE_FILE "mnist/t10k-images.idx3-ubyte"
#define MNIST_TEST_LABEL_FILE "mnist/t10k-labels.idx1-ubyte"

typedef struct {
    uint32_t size_training;
    uint32_t size_test;
    uint8_t * training_labels;
    uint8_t * test_labels;
    Matrix * training_images;
    Matrix * test_images;
} MNISTData;

MNISTData mnist_new();

void mnist_print_image(const Matrix * p_matrix);

void mnist_delete(MNISTData * p_data);

#endif
