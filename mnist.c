#include "mnist.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#ifdef DEBUG
#include <assert.h>
#endif

#ifdef LITTLE_ENDIAN
static void swap_endian(void * p_data, int size){
    uint8_t * p_bytes = (uint8_t *)(p_data);
    for(int i = 0; i < size / 2; i++){
        uint8_t swap = p_bytes[i];
        p_bytes[i] = p_bytes[size - 1 - i];
        p_bytes[size - 1 - i] = swap;
    }
}
#endif

static void read_idx_header(FILE * fd, int * p_size){
#ifdef DEBUG
    uint32_t magic;
    fread((void *)&magic, 1, 4, fd);
#ifdef LITTLE_ENDIAN
    swap_endian((void *)&magic, 4);
#endif
    printf("Read magic number 0x%x (%u)\n", magic, magic);
#endif

    uint32_t size;
    fread((void *)&size, 1, 4, fd);
#ifdef LITTLE_ENDIAN
    swap_endian((void *)&size, 4);
#endif
    *p_size = size;
}

static void read_idx1(FILE * fd, int * p_size, int ** pp_data){
    read_idx_header(fd, p_size);

    *pp_data = (int *)malloc((*p_size) * sizeof(int));
    for(int i = 0; i < (int)*p_size; i++){
        uint8_t label;
        fread((void *)(&label), 1, 1, fd);
        (*pp_data)[i] = label;
    }
}

static void read_idx3(FILE * fd, int * p_size, Matrix ** pp_data){
    read_idx_header(fd, p_size);

    *pp_data = (Matrix *)malloc((*p_size) * sizeof(Matrix));
    for(int i = 0; i < (int)*p_size; i++){
        // Parse matrices
    }
}

MNISTData mnist_new(){
    MNISTData data;
    data.size_training = 0;
    data.size_test = 0;

    FILE * fd_train_images = fopen(MNIST_TRAINING_IMAGE_FILE, "r");
    FILE * fd_train_labels = fopen(MNIST_TRAINING_LABEL_FILE, "r");
    FILE * fd_test_images = fopen(MNIST_TEST_IMAGE_FILE, "r");
    FILE * fd_test_labels = fopen(MNIST_TEST_LABEL_FILE, "r");
#ifdef DEBUG
    assert(fd_train_images != NULL);
    assert(fd_train_labels != NULL);
    assert(fd_test_images != NULL);
    assert(fd_test_labels != NULL);
#endif

    read_idx1(fd_train_labels, &data.size_training, &data.training_labels);
    read_idx1(fd_test_labels, &data.size_test, &data.test_labels);
    read_idx3(fd_train_images, &data.size_training, &data.training_images);
    read_idx3(fd_train_images, &data.size_test, &data.test_images);

    fclose(fd_train_images);
    fclose(fd_train_labels);
    fclose(fd_test_images);
    fclose(fd_test_labels);

    return data;
}

void mnist_delete(MNISTData * p_data){
    free(p_data->training_labels);
    free(p_data->test_labels);
    free(p_data->training_images);
    free(p_data->test_images);
    p_data->size_training = 0;
    p_data->size_test = 0;
}
