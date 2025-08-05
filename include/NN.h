#ifndef __NN_H_
#define __NN_H_

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#define NN_DEBUG_PRINT 1
#define NN_INIT_ZERO 1
#define NN_MEMORY_TRIM_AFTER_FREE 1

typedef enum {
    AUTO,
    GPU,
    GPU_CUDA,
    CPU,
    TPU,
} NN_device;

typedef enum {
    RELU,
    SIGMOID,
    SOFTMAX,
    TANH,
} NN_activation_function;

typedef enum {
    ADAM,
    GRADIENT_DESCENT,
    STOCHASTIC_GRADIENT_DESCENT,
    MINI_BATCH_GRADIENT_DESCENT,
} NN_optimizer;

typedef enum {
    MSE,
} NN_loss_function;


typedef struct {
    float learning_rate;
    NN_optimizer optimizer;
    bool use_batching;
} NN_learning_settings;


typedef struct {
    NN_activation_function activation;
    NN_device device_type;
} NN_use_settings;

typedef struct {
    float* in;
    float** out;

    unsigned int* neurons_per_layer;
    unsigned int layers;

    float*** weights;
    float** bias;
} NN_network;

typedef struct {
    NN_use_settings* settings;
    char* device_name;
    NN_network* network;
} NN_processor;

typedef struct {
    NN_learning_settings* learning_settings;
    NN_processor processor;
    float ***grad_weights; // same shape as weights
    float **grad_bias;     // same shape as bias
} NN_trainer;


// init functions
NN_network* NN_network_init(unsigned int* neurons_per_layer,
                            unsigned int layers);
NN_network* NN_network_init_from_file(char* filepath);
NN_trainer* NN_trainer_init(NN_network* network, NN_learning_settings* learn_settings, NN_use_settings* use_settings, char* device_name);
NN_processor* NN_processor_init(NN_network* network, NN_use_settings* settings, char* device_name);

// free functions
void NN_network_free(NN_network* network);
void NN_trainer_free(NN_trainer* trainer);
void NN_processor_free(NN_processor* processor);

// use functions
void NN_trainer_train(NN_trainer* trainer, float* in, float* desired_out);
void NN_trainer_accumulate(NN_trainer *trainer, float *input, float *target);
void NN_trainer_apply(NN_trainer *trainer, unsigned int batch_size);
void NN_processor_process(NN_processor* processor, float* in, float* out);

// utility functions
float NN_trainer_loss(NN_trainer* trainer, float* desired);
void NN_network_save_to_file(NN_network* network, char* filepath);



/*
    OLD CODE



typedef struct {
    float* weights;
    float* bias;
    unsigned int size;
} NN_layer;

typedef struct {
    NN_layer* layers;
    unsigned int n_layers;
    unsigned int n_neurons;
} NN_network;

NN_layer NN_layer_init(float* weights, float* bias, unsigned int size);
void NN_layer_destroy(NN_layer* layer);

void NN_layer_process(float* input, NN_layer* layer1, NN_layer* layer2, float* output);


NN_network NN_network_init(NN_layer* layers, unsigned int n_layers, unsigned int n_neurons);
void NN_network_destroy(NN_network* network);



float* NN_network_process(NN_network* network, float* output, float* input);
float* NN_network_forward(NN_network* network, float* output, float* input);
void NN_network_backprop(NN_network* network, float* real_output, float* desired_output, float* intermediates, float learning_rate);
float NN_network_loss(NN_network* network, float *real_output, float *desired_output);
*/
#endif