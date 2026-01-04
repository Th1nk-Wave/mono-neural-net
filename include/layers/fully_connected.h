#ifndef __NN_FULLY_CONNECTED_H_
#define __NN_FULLY_CONNECTED_H_

#include "NN.h"
#include <stdio.h>

// fully connected
typedef struct {
    float** grad_weights;
    float* grad_bias;
} NN_layer_fully_connected_grads_buf;



typedef struct {
    float* bias;
    float** weights;
} NN_layer_fully_connected_params;



void NN_fully_connected_forward(NN_layer* layer, const float* input);
void NN_fully_connected_randomise(NN_layer* layer, float min, float max);

void NN_fully_connected_backward(NN_training_layer* layer, const float* input, const float* delta_next, float* delta_out);
void NN_fully_connected_apply(NN_training_layer* layer, unsigned int batch_size);

NN_API NN_layer* NN_create_fully_connected_layer(unsigned int n_in, unsigned int n_out, NN_activation_function activation);
void NN_clean_up_fully_connected_layer(NN_layer *layer);

void NN_set_fully_connected_training_layer(NN_training_layer* layer, NN_learning_settings* settings);
void NN_clean_up_fully_connected_training_layer(NN_training_layer* layer);

NN_API int NN_fully_connected_save_to_file(NN_layer* layer, FILE* f);
NN_API NN_layer* NN_fully_connected_init_from_file(FILE* f);

#endif