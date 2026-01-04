#include "NN.h"
#include "RNG.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ucontext.h>

#if NN_DEBUG_PRINT
#if defined(__linux__)
// https://linux.die.net/man/3/malloc_usable_size
#include <malloc.h>
size_t malloc_size(const void *p) {
    return malloc_usable_size((void*)p);
}
#elif defined(__APPLE__)
// https://www.unix.com/man-page/osx/3/malloc_size/
#include <malloc/malloc.h>
size_t malloc_size(const void *p) {
    return malloc_size(p);
}
#elif defined(_WIN32)
// https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/msize
#include <malloc.h>
size_t malloc_size(const void *p) {
    return _msize((void *)p);
}
#else
#error "system not recognised when determining malloc_size debug function, you may be able to fix this by disabling the NN_DEBUG_PRINT flag or adding your own definitions"
#endif
#endif

#include "layers/recurrent.h"
#include "layers/fully_connected.h"


// DERIV FUNCTIONS
inline float NN_loss_deriv(NN_loss_function function, float out, float target) {
    switch (function) {
        case MSE: return out - target;
        case MAE: return (out > target) ? 1.0f : -1.0f;
        case HUBER: {
            float diff = out - target;
            const float delta = 1.0f;
            if (fabsf(diff) <= delta) return diff;
            return (diff > 0) ? delta : -delta;
        }
        // cross entropy only valid if activation is SOFTMAX
        case CROSS_ENTROPY: return out - target;
        default: return out - target;
    }
}

inline float NN_activation_deriv(NN_activation_function activation, float in) {
    float deriv = 1.0f;
    switch (activation) {
        case SIGMOID:   deriv = in * (1.0f - in); break;
        case TANH:      deriv = 1.0f - in * in; break;
        case RELU:      deriv = (in > 0) ? 1.0f : 0.0f; break;
        case LERELU:    deriv = (in > 0) ? 1.0f : LERELU_FACTOR; break;
        default: break;
    }
    return deriv;
}

// INTERNAL
inline float NN_apply_activation(NN_activation_function activation, float in) {
    switch (activation) {
        default:
        case RELU:    return in > 0 ? in : 0; break;
        case LERELU:  return in > 0 ? in : in * LERELU_FACTOR; break;
        case SIGMOID: return 1.0f / (1.0f + expf(-in)); break;
        case TANH:    return tanhf(in); break;
    }
}


// LAYER IMPLEMENTATION
NN_layer* NN_layer_init(unsigned int n_in, unsigned int n_out, NN_activation_function activation) {
    NN_layer* layer = (NN_layer*)malloc(sizeof(NN_layer));
    layer->activation = activation;
    layer->in_size = n_in;
    layer->out_size = n_out;
    layer->out = (float*)malloc(sizeof(float)*n_out);
    layer->type = NULL_LAYER;
    return layer;
}

void NN_layer_free(NN_layer* layer) {
    switch (layer->type) {
        // fully connected
        case FULLY_CONNECTED: NN_clean_up_fully_connected_layer(layer); break;
        case RECURRENT: NN_clean_up_rnn_layer(layer); break;



        // fallback error
        case NULL_LAYER:
        default: {
            break;
        }
    }
    free(layer->out);
    free(layer);
}


// NETWORK IMPLEMENTATION
NN_network* NN_network_init(NN_layer** layers, unsigned int n_layers) {
    NN_network* net = (NN_network*)malloc(sizeof(NN_network));
    net->layers = layers;
    net->n_layers = n_layers;
    return net;
}

void NN_network_free(NN_network *network) {
    for (unsigned int layer = 0; layer < network->n_layers; layer++) {
        NN_layer_free(network->layers[layer]);
    }
    free(network);
}


// PROCESSOR IMPLEMENTATION
NN_processor* NN_processor_init(NN_network* network, NN_use_settings* settings, char* device_name) {
    unsigned int _size = strlen(device_name) + 1;

    NN_processor* processor = (NN_processor*)malloc(sizeof(NN_processor));
    processor->device_name = malloc(_size);
    for (unsigned int i = 0; i < _size; i++) {
        processor->device_name[i] = device_name[i];
    }

    processor->settings = settings;
    processor->network = network;

    return processor;
}

void NN_processor_free(NN_processor* processor) {
    free(processor->device_name);
    free(processor);
}

void NN_processor_process(NN_processor* processor, float* in, float* out) {
    NN_network* net = processor->network;
    unsigned int n_layers = net->n_layers;


    net->layers[0]->forward(net->layers[0],in);
    for (unsigned int layer = 1; layer < n_layers; layer++) {
        net->layers[layer]->forward(net->layers[layer],net->layers[layer-1]->out);
    }
    
    for (unsigned int i = 0; i < net->layers[n_layers-1]->out_size; i++) {
        out[i] = net->layers[n_layers-1]->out[i];
    }
}


// TRAINING LAYER IMPLEMENTATION
NN_training_layer* NN_training_layer_init(NN_layer* base, NN_learning_settings* settings) {
    NN_training_layer* layer = (NN_training_layer*)malloc(sizeof(NN_training_layer));
    layer->base = base;
    layer->settings = settings;

    switch (settings->optimizer) {
        case ADAMW:
        case ADAM: {
            NN_ADAM_buf* buf = (NN_ADAM_buf*)malloc(sizeof(NN_ADAM_buf));
            buf->adam_t = 0;
            buf->m_bias = calloc(base->out_size, sizeof(float));
            buf->v_bias = calloc(base->out_size, sizeof(float));
            buf->m_weights = (float**)malloc(sizeof(float*)*base->out_size);
            buf->v_weights = (float**)malloc(sizeof(float*)*base->out_size);
            for (unsigned int i = 0; i < base->out_size; i++) {
                buf->m_weights[i] = calloc(base->in_size, sizeof(float));
                buf->v_weights[i] = calloc(base->in_size, sizeof(float));
            }
            layer->optimizer_buf = buf;
            break;
        }
        default: {
            break;
        }
    }
    switch (base->type) {
        case FULLY_CONNECTED: NN_set_fully_connected_training_layer(layer,settings); break;
        case RECURRENT: NN_set_rnn_training_layer(layer, settings); break;
        default: break;
    }
    return layer;
}

void NN_training_layer_free(NN_training_layer* layer) {
    switch (layer->settings->optimizer) {
        case ADAMW:
        case ADAM: {
            NN_ADAM_buf* buf = (NN_ADAM_buf*)layer->optimizer_buf;
            free(buf->m_bias);
            free(buf->v_bias);

            for (unsigned int i = 0; i < layer->base->out_size; i++) {
                free(buf->m_weights[i]);
                free(buf->v_weights[i]);
            }

            free(buf->m_weights);
            free(buf->v_weights);
            break;
        }
        default: {
            break;
        }
    }
    switch (layer->base->type) {
        case FULLY_CONNECTED: NN_clean_up_fully_connected_training_layer(layer); break;
        case RECURRENT: NN_clean_up_rnn_training_layer(layer); break;
        default: break;
    }
    free(layer);
}


// TRAINER IMPLEMENTATION
NN_trainer* NN_trainer_init(NN_processor* processor, NN_learning_settings* learning_settings) {
    NN_trainer* trainer = (NN_trainer*)malloc(sizeof(NN_trainer));
    trainer->learning_settings = learning_settings;
    trainer->processor = processor;

    trainer->training_layers = (NN_training_layer**)malloc(sizeof(NN_training_layer*)*processor->network->n_layers);
    for (unsigned int i = 0; i < processor->network->n_layers; i++) {
        trainer->training_layers[i] = NN_training_layer_init(processor->network->layers[i], learning_settings);
    }
    return trainer;
}

void NN_trainer_free(NN_trainer *trainer) {
    for (unsigned int i = 0; i < trainer->processor->network->n_layers; i++) {
        NN_training_layer_free(trainer->training_layers[i]);
    }
    free(trainer->training_layers);
    free(trainer);
}

void NN_trainer_accumulate(NN_trainer *trainer, float *input, float *target) {
    if (!trainer->learning_settings->use_batching) { fprintf(stderr, "ERROR: can't accumulate to gradient buffers because they have not been allocated\n    FIX: enable use_batching in learning_settings, this will roughly double memory usage though as a result");}
    NN_network* net = trainer->processor->network;
    NN_use_settings* settings = trainer->processor->settings;
    unsigned int n_layers = net->n_layers;

    // run forward pass
    NN_processor_process(trainer->processor, input, net->layers[n_layers-1]->out);

    float* delta = malloc(sizeof(float) * net->layers[n_layers-1]->out_size);

    // output layer delta
    for (unsigned int i = 0; i < net->layers[n_layers-1]->out_size; i++) {
        delta[i] =
            NN_loss_deriv(trainer->learning_settings->loss_function,
                      net->layers[n_layers-1]->out[i],
                      target[i]);
    }

    // backward pass
    for (int l = n_layers - 1; l >= 0; l--) {
        float *prev_delta = (l == 0) ? malloc(sizeof(float) * net->layers[l]->in_size) : malloc(sizeof(float) * net->layers[l - 1]->out_size);
        float* layer_input = (l == 0) ? input : net->layers[l - 1]->out;

        trainer->training_layers[l]->backward(
            trainer->training_layers[l],
            layer_input,
            delta,
            prev_delta
        );


        free(delta);
        delta = prev_delta;
    }

    free(delta);
    
}

void NN_trainer_apply(NN_trainer *trainer, unsigned int batch_size) {
    NN_network* net = trainer->processor->network;

    for (unsigned int l = 0; l < net->n_layers; l++) {
        NN_training_layer* tl = trainer->training_layers[l];

        if (tl->apply) {
            tl->apply(tl, batch_size);
        }
    }
}

float NN_trainer_loss(NN_trainer *trainer, float *desired) {
    NN_network* net = trainer->processor->network;
    float loss = 0.0f;
    unsigned int last = net->n_layers - 1;
    unsigned int out_n = net->layers[last]->out_size;
    for (unsigned int i = 0; i < out_n; i++) {
        float diff = NN_loss_deriv(trainer->learning_settings->loss_function, net->layers[last]->out[i], desired[i]);
        loss += diff * diff;
    }
    return loss / out_n;
}


void NN_network_randomise_xaivier(NN_network *net, float weight_min, float weight_max) {
    for (unsigned int l = 0; l < net->n_layers; l++) {
        net->layers[l]->randomize(net->layers[l],weight_min,weight_max);
    }
}




int NN_network_save_to_file(NN_network *network, char *filepath) {
    FILE *f = fopen(filepath, "wb");
    if (!f) return -1;

    uint16_t version = NN_FILE_VERSION;
    uint8_t activation_id = 0;

    if (fwrite(&version,                sizeof(uint16_t),       1, f) != 1) goto error;   
    if (fwrite(&network->n_layers,      sizeof(uint16_t),       1, f) != 1) goto error;

    for (unsigned int l = 0; l < network->n_layers; l++) {
        switch (network->layers[l]->type) {
            case FULLY_CONNECTED:   NN_fully_connected_save_to_file(network->layers[l], f); break;
            case RECURRENT:         NN_rnn_save_to_file(network->layers[l], f); break;
            default:                perror("network type id %u has no file save handler\n"); break;
        }
    }


    fclose(f);
    return 1;

error:
    printf("[ERROR] error in NN_network_save_to_file\n");
    fclose(f);
    return -1;
}

NN_network* NN_network_init_from_file(char *filepath) {
    FILE *f = fopen(filepath, "rb");
    if (!f) return 0; // no file

    uint16_t version;
    uint16_t n_layers;

    // version check
    if (fread(&version, sizeof(uint16_t), 1, f) != 1) goto error;           // [2 byte] version
    if (version != NN_FILE_VERSION) {
        fprintf(stderr, "unsupported .net file version: %u\n", version);
        goto error;
    }

    if (fread(&n_layers, sizeof(uint16_t), 1, f) != 1) goto error;          // [2 byte] layers

    NN_layer** layers = malloc(sizeof(NN_layer*) * n_layers);
    for (unsigned int l = 0; l < n_layers; l++) {
        NN_layer_type type;
        if (fread(&type, sizeof(NN_layer_type), 1, f) != 1) goto error;     // [1 byte] layer header type
        switch (type) {
            case FULLY_CONNECTED: layers[l] = NN_fully_connected_init_from_file(f); break;
            case RECURRENT: layers[l] = NN_rnn_init_from_file(f); break;
            default: fprintf(stderr,"network type id %u has no file save handler\n",type); break;
        }
    }

    NN_network* net = NN_network_init(layers, n_layers);
    if (!net) goto error;
    fclose(f);
    return  net;

error:
    printf("[ERROR] error in NN_network_init_from_file\n");
    fclose(f);
    return NULL;
}