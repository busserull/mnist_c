#include "mnist.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#ifdef DEBUG
#include <assert.h>
#endif

#define IMAGE_THRESHOLD 127

static uint32_t read_size(FILE * fd, int dimension){
    fseek(fd, dimension * 4 + 4, SEEK_SET);
    uint8_t size_bytes[4];
    fread((void *)size_bytes, 1, 4, fd);

    uint32_t size = 0x00000000
        | (size_bytes[0] << 24)
        | (size_bytes[1] << 16)
        | (size_bytes[2] << 8)
        | (size_bytes[3] << 0)
        ;

    return size;
}

static void skip_header(FILE * fd){
    fseek(fd, 3, SEEK_SET);
    uint8_t dimension;
    fread((void *)&dimension, 1, 1, fd);
    fseek(fd, 4 + dimension * 4, SEEK_SET);
}

static MnistSet make_set(FILE * fd_labels, FILE * fd_images){
#ifdef DEBUG
    assert(fd_labels != NULL);
    assert(fd_images != NULL);
#endif

    uint32_t labels_size = read_size(fd_labels, 0);
#ifdef DEBUG
    uint32_t images_size = read_size(fd_images, 0);
    assert(labels_size == images_size);
#endif
    uint32_t rows = read_size(fd_images, 1);
    uint32_t columns = read_size(fd_images, 2);

    MnistSet set;
    set.size = labels_size;
    set.labels = (MnistLabel *)malloc(set.size * sizeof(MnistLabel));
    set.images = (MnistImage *)malloc(set.size * sizeof(MnistImage));

    skip_header(fd_labels);
    fread((void *)set.labels, 1, set.size, fd_labels);

    skip_header(fd_images);
    for(uint32_t i = 0; i < set.size; i++){
        set.images[i].rows = rows;
        set.images[i].columns = columns;
        set.images[i].points = (uint8_t *)malloc(rows * columns);
        fread((void *)set.images[i].points, 1, rows * columns, fd_images);
    }

    return set;
}

void mnist_new(MnistSet * p_training_set, MnistSet * p_test_set){
    FILE * fd_training_labels = fopen(MNIST_TRAINING_LABELS, "r");
    FILE * fd_training_images = fopen(MNIST_TRAINING_IMAGES, "r");
    *p_training_set = make_set(fd_training_labels, fd_training_images);

    FILE * fd_test_labels = fopen(MNIST_TEST_LABELS, "r");
    FILE * fd_test_images = fopen(MNIST_TEST_IMAGES, "r");
    *p_test_set = make_set(fd_test_labels, fd_test_images);

    fclose(fd_training_labels);
    fclose(fd_training_images);
    fclose(fd_test_labels);
    fclose(fd_test_images);
}

Matrix mnist_vectorize_image(const MnistImage image){
    int rows = image.rows;
    int columns = image.columns;

    Matrix vector = matrix_new(rows * columns, 1);

    for(int x = 0; x < rows; x++){
        for(int y = 0; y < columns; y++){
            double intensity = image.points[x * columns + y];
            matrix_set(vector, x * columns + y, 0, intensity);
        }
    }

    return vector;
}

void mnist_print_image(const MnistImage image){
    for(int x = 0; x < image.rows; x++){
        for(int y = 0; y < image.columns; y++){
            if(image.points[x * image.columns + y] > IMAGE_THRESHOLD){
                printf("x");
            }
            else{
                printf(" ");
            }
        }
        printf("\n");
    }
}

void mnist_delete(MnistSet * p_data_set){
    free(p_data_set->labels);
    for(int i = 0; i < p_data_set->size; i++){
        free(p_data_set->images[i].points);
    }
    free(p_data_set->images);
    p_data_set->size = 0;
}
