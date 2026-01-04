#ifndef __NN_H_
#define __NN_H_

#include <stdbool.h>
#include <stdint.h>

#define NN_FILE_VERSION 2
#define NN_VERSION "0.2.0"

#define NN_DEBUG_PRINT 1
#define NN_INIT_ZERO 1
#define NN_MEMORY_TRIM_AFTER_FREE 1

#define LERELU_FACTOR 0.1



typedef enum {
    AUTO,
    GPU,
    GPU_CUDA,
    CPU,
    TPU,
} NN_device;

typedef enum:char {
    RELU,
    SIGMOID,
    SOFTMAX,
    TANH,
    LERELU,
} NN_activation_function;

typedef enum {
    ADAM,
    ADAMW,
    GRADIENT_DESCENT,
    STOCHASTIC_GRADIENT_DESCENT,
    MINI_BATCH_GRADIENT_DESCENT,
} NN_optimizer;

typedef enum {
    MSE,
    CROSS_ENTROPY,
    MAE,
    HUBER,
} NN_loss_function;

typedef enum:char {
    FULLY_CONNECTED,
    CONVOLUTIONAL,
    RECURRENT,
    POOLING,
    DROPOUT,
    DECONVOLUTIONAL,
    ATTENTION,
    BATCH_NORM,
    SKIP,
    GATED_RECURRENT_UNIT,
    LONG_SHORT_TERM_MEMORY,
    NULL_LAYER,
} NN_layer_type;


typedef struct {
    float learning_rate;
    NN_optimizer optimizer;
    NN_loss_function loss_function;
    bool use_batching;

    float weight_decay;

    // adam
    float adam_beta1;
    float adam_beta2;
    float adam_epsilon;
} NN_learning_settings;

typedef struct {
    NN_device device_type;
} NN_use_settings;

typedef struct NN_layer NN_layer;
struct NN_layer {
    // shape info
    uint32_t in_size, out_size;
    NN_layer_type type;
    NN_activation_function activation;

    // forward pass output buffer
    float* out;

    // layer specific 
    void* params;

    // vtable
    void (*forward)(struct NN_layer* layer, const float* input);
    void (*randomize)(struct NN_layer* layer, float min, float max);
};

typedef  struct NN_training_layer NN_training_layer ;
struct NN_training_layer {
    NN_layer* base;                 // base layer (used by backprop)
    NN_learning_settings* settings;  // learn settings (used by backprop and weight apply)
    void *grads;                    // gradient buffers (optional)
    void *optimizer_buf;            // ADAM or other optimizer states

    // vtable
    void (*backward)(struct NN_training_layer* layer, const float* input, const float* delta_next, float* delta_out);
    void (*apply)(struct NN_training_layer* layer, unsigned int batch_size);
};

typedef struct {
    float** m_weights;
    float** v_weights;
    float* m_bias;
    float* v_bias;
    unsigned int adam_t;   // time step for bias correction
} NN_ADAM_buf;




typedef struct {
    NN_layer** layers;
    unsigned int n_layers;
} NN_network;

typedef struct {
    NN_use_settings* settings;
    char* device_name;
    NN_network* network;
} NN_processor;

typedef struct {
    NN_learning_settings* learning_settings;
    NN_training_layer** training_layers;
    NN_processor* processor;
} NN_trainer;


// init functions
NN_network* NN_network_init(NN_layer** layers,
                            unsigned int n_layers);
NN_network* NN_network_init_from_file(char* filepath);
NN_trainer* NN_trainer_init(NN_processor* processor, NN_learning_settings* learning_settings);
NN_processor* NN_processor_init(NN_network* network, NN_use_settings* settings, char* device_name);
NN_layer* NN_layer_init(unsigned int n_in, unsigned int n_out, NN_activation_function activation);
NN_training_layer* NN_training_layer_init(NN_layer* base, NN_learning_settings* learning_settings);

// free functions
void NN_network_free(NN_network* network);
void NN_trainer_free(NN_trainer* trainer);
void NN_processor_free(NN_processor* processor);
void NN_layer_free(NN_layer* layer);

// use functions
void NN_trainer_train(NN_trainer* trainer, float* in, float* desired_out);
void NN_trainer_accumulate(NN_trainer *trainer, float *input, float *target);
void NN_trainer_apply(NN_trainer *trainer, unsigned int batch_size);
void NN_processor_process(NN_processor* processor, float* in, float* out);

// utility functions
void NN_network_randomise_xaivier(NN_network* network, float weight_min, float weight_max);
float NN_trainer_loss(NN_trainer* trainer, float* desired);
NN_network* NN_network_init_from_file(char* filepath);
int NN_network_save_to_file(NN_network* network, char* filepath);

// internal
float NN_apply_activation(NN_activation_function activation, float in);

// deriv functions
float NN_loss_deriv(NN_loss_function loss, float out, float target);
float NN_activation_deriv(NN_activation_function activation, float in);

// randomize
void NN_network_randomise_xaivier(NN_network *net, float weight_min, float weight_max);

#endif