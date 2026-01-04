#ifndef __NN_RECURRENT_H_
#define __NN_RECURRENT_H_

#include "NN.h"
#include <stdio.h>

// RNN parameters
typedef struct {
    float** weights_input;   // input -> hidden
    float** weights_hidden;  // hidden -> layer out
    float* bias;
    float* hidden_state;     // current hidden state
} NN_layer_rnn_params;

typedef struct {
    float** grad_weights_input;
    float** grad_weights_hidden;
    float* grad_bias;
    float* grad_hidden_state; // used for truncated BPTT
} NN_layer_rnn_grads_buf;

void NN_rnn_forward(NN_layer* layer, const float* input);
void NN_rnn_randomise(NN_layer* layer, float min, float max);

void NN_rnn_backward(NN_training_layer* layer, const float* input, const float* delta_next, float* delta_out);
void NN_rnn_apply(NN_training_layer* layer, unsigned int batch_size);

NN_API NN_layer* NN_create_rnn_layer(unsigned int n_in, unsigned int n_out, NN_activation_function activation);
void NN_clean_up_rnn_layer(NN_layer* layer);

void NN_set_rnn_training_layer(NN_training_layer* layer, NN_learning_settings* settings);
void NN_clean_up_rnn_training_layer(NN_training_layer* layer);

NN_API int NN_rnn_save_to_file(NN_layer* layer, FILE* f);
NN_API NN_layer* NN_rnn_init_from_file(FILE* f);

#endif