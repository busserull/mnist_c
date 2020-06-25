#include "mnist.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#ifdef DEBUG
#include <assert.h>
#endif

#define PRINT_IMAGE_THRESHOLD 127

typedef struct {
    uint8_t data_bytes;
    uint8_t dimension;
    uint32_t * sizes;
} IDXHeader;

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

static IDXHeader read_idx_header(FILE * fd){
    uint8_t magic[4];
    fread((void *)magic, 1, 4, fd);

    IDXHeader header;
    header.dimension = magic[3];

    switch(magic[2]){
        case 0x08:
        case 0x09:
            header.data_bytes = 1;
            break;
        case 0x0b:
            header.data_bytes = 2;
            break;
        case 0x0c:
        case 0x0d:
            header.data_bytes = 4;
            break;
        case 0x0e:
            header.data_bytes = 8;
            break;
        default:
            fprintf(stderr, "Unknown IDX data type\n");
            exit(1);
    }

    header.sizes = (uint32_t *)malloc(header.dimension * sizeof(uint32_t));
    for(int i = 0; i < header.dimension; i++){
        uint32_t size;
        fread((void *)&size, 1, 4, fd);
#ifdef LITTLE_ENDIAN
        swap_endian((void *)&size, 4);
#endif
        header.sizes[i] = size;
    }

    return header;
}

static void read_labels(FILE * fd, uint32_t * p_size, uint8_t ** pp_labels){
    IDXHeader header = read_idx_header(fd);
#ifdef DEBUG
    assert(header.dimension == 1);
#endif
    *p_size = header.sizes[0];

    *pp_labels = (uint8_t *)malloc((*p_size) * sizeof(uint8_t));
    for(int i = 0; i < *p_size; i++){
        fread((*pp_labels) + i, 1, 1, fd);
    }

    free(header.sizes);
}

static void read_images(FILE * fd, Matrix ** pp_images){
    IDXHeader header = read_idx_header(fd);
#ifdef DEBUG
    assert(header.dimension == 3);
#endif
    uint32_t images = header.sizes[0];
    uint32_t rows = header.sizes[1];
    uint32_t columns = header.sizes[2];

    *pp_images = (Matrix *)malloc(images * sizeof(Matrix));
    for(uint32_t i = 0; i < images; i++){
        (*pp_images)[i] = matrix_new(rows, columns);

        for(uint32_t r = 0; r < rows; r++){
            for(uint32_t c = 0; c < columns; c++){
                uint8_t intensity;
                fread((void *)&intensity, 1, 1, fd);
                matrix_set((*pp_images) + i, r, c, (double)intensity);
            }
        }
    }

    free(header.sizes);
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

    read_labels(fd_train_labels, &data.size_training, &data.training_labels);
    read_labels(fd_test_labels, &data.size_test, &data.test_labels);
    read_images(fd_train_images, &data.training_images);
    read_images(fd_test_images, &data.test_images);

    fclose(fd_train_images);
    fclose(fd_train_labels);
    fclose(fd_test_images);
    fclose(fd_test_labels);

    return data;
}

void mnist_print_image(const Matrix * p_matrix){
    for(int x = 0; x < p_matrix->x; x++){
        for(int y = 0; y < p_matrix->y; y++){
            if(matrix_get(p_matrix, x, y) > PRINT_IMAGE_THRESHOLD){
                printf("x");
            }
            else{
                printf(" ");
            }
        }
        printf("\n");
    }
}

void mnist_delete(MNISTData * p_data){
    free(p_data->training_labels);
    free(p_data->test_labels);
    for(int i = 0; i < p_data->size_training; i++){
        matrix_delete(p_data->training_images + i);
    }
    for(int i = 0; i < p_data->size_test; i++){
        matrix_delete(p_data->test_images + i);
    }
    free(p_data->training_images);
    free(p_data->test_images);
    p_data->size_training = 0;
    p_data->size_test = 0;
}
