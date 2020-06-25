#ifndef MNIST_H
#define MNIST_H
#include "matrix.h"
#include <stdint.h>

#define MNIST_TRAINING_IMAGES "mnist/train-images.idx3-ubyte"
#define MNIST_TRAINING_LABELS "mnist/train-labels.idx1-ubyte"
#define MNIST_TEST_IMAGES "mnist/t10k-images.idx3-ubyte"
#define MNIST_TEST_LABELS "mnist/t10k-labels.idx1-ubyte"

typedef uint8_t MnistLabel;

typedef struct {
    uint8_t rows;
    uint8_t columns;
    uint8_t * points;
} MnistImage;

typedef struct {
    uint32_t size;
    MnistLabel * labels;
    MnistImage * images;
} MnistSet;

void mnist_new(MnistSet * p_training_set, MnistSet * p_test_set);

Matrix mnist_vectorize_image(const MnistImage * p_image);

void mnist_print_image(const MnistImage * p_image);

void mnist_delete(MnistSet * p_data_set);

#endif
