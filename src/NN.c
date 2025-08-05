#include "NN.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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




NN_network* NN_network_init(unsigned int* neurons_per_layer,
                            unsigned int layers) {
    NN_network* net = (NN_network*)malloc(sizeof(NN_network));
    net->layers = layers;
    net->neurons_per_layer = malloc(sizeof(unsigned int) * layers);
    for (unsigned int layer = 0; layer < layers; layer++) {
        net->neurons_per_layer[layer] = neurons_per_layer[layer];
    }

    net->weights = malloc(sizeof(float**) * (layers - 1));
    net->bias = malloc(sizeof(float*) * (layers - 1));

    for (unsigned int layer = 0; layer < layers - 1; layer++) {
        unsigned int in_n = neurons_per_layer[layer];
        unsigned int out_n = neurons_per_layer[layer + 1];

        net->weights[layer] = malloc(sizeof(float*) * in_n);
        net->bias[layer] = malloc(sizeof(float) * out_n);

        for (unsigned int i = 0; i < in_n; i++) {
            net->weights[layer][i] = malloc(sizeof(float) * out_n);
            #if NN_INIT_ZERO
                for (unsigned int j = 0; j < out_n; j++) {
                    net->weights[layer][i][j] = 0.0f;
                }
            #endif
        }

        #if NN_INIT_ZERO
            for (unsigned int j = 0; j < out_n; j++) {
                net->bias[layer][j] = 0.0f;
            }
        #endif
    }

    net->in = malloc(sizeof(float) * neurons_per_layer[0]);
    net->out = malloc(sizeof(float*) * layers);
    for (unsigned int layer = 0; layer < layers; layer++) {
        net->out[layer] = malloc(sizeof(float) * neurons_per_layer[layer]);
        #if NN_INIT_ZERO
            for (unsigned int i = 0; i < neurons_per_layer[layer]; i++) {
                net->out[layer][i] = 0.0f;
            }
        #endif
    }

    #if NN_DEBUG_PRINT
    size_t neuron_layer_alloc = sizeof(unsigned int) * layers;
    size_t in_alloc = sizeof(float) * neurons_per_layer[0];

    size_t weights_alloc = sizeof(float**) * (layers - 1);
    size_t bias_alloc = sizeof(float*) * (layers - 1);

    for (unsigned int layer = 0; layer < layers - 1; layer++) {
        weights_alloc += sizeof(float*) * (neurons_per_layer[layer]); // each weight row
        weights_alloc += sizeof(float*) * 1; // extra pointer for bias connection?
        bias_alloc += sizeof(float) * (neurons_per_layer[layer + 1]);

        weights_alloc += sizeof(float) * (neurons_per_layer[layer + 1]) * (neurons_per_layer[layer] + 1);
    }

    size_t out_alloc = sizeof(float*) * layers;
    for (unsigned int layer = 0; layer < layers; layer++) {
        out_alloc += sizeof(float) * neurons_per_layer[layer];
    }

    size_t total = neuron_layer_alloc + in_alloc + weights_alloc + bias_alloc + out_alloc;

    printf("allocated %.6f MiB of memory for neural network\n", total / (1024.0 * 1024.0));
    printf("    L neuron_layer_info: %.6f MiB (%.2f%%)\n", neuron_layer_alloc / (1024.0 * 1024.0), 100.0 * neuron_layer_alloc / total);
    printf("    L in_neurons: %.6f MiB (%.2f%%)\n", in_alloc / (1024.0 * 1024.0), 100.0 * in_alloc / total);
    printf("    L out_neurons: %.6f MiB (%.2f%%)\n", out_alloc / (1024.0 * 1024.0), 100.0 * out_alloc / total);
    printf("    L weights: %.6f MiB (%.2f%%)\n", weights_alloc / (1024.0 * 1024.0), 100.0 * weights_alloc / total);
    printf("    L bias: %.6f MiB (%.2f%%)\n", bias_alloc / (1024.0 * 1024.0), 100.0 * bias_alloc / total);
    #endif

    return net;
}

void NN_network_free(NN_network *net) {
    #if NN_DEBUG_PRINT
        size_t total_freed = 0;
    #endif

    for (unsigned int layer = 0; layer < net->layers - 1; layer++) {
        unsigned int in_n = net->neurons_per_layer[layer];

        for (unsigned int i = 0; i < in_n; i++) {
            #if NN_DEBUG_PRINT
                total_freed += malloc_size(net->weights[layer][i]);
            #endif
            free(net->weights[layer][i]);
        }
        #if NN_DEBUG_PRINT
            total_freed += malloc_size(net->weights[layer]);
            total_freed += malloc_size(net->bias[layer]);
        #endif
        free(net->weights[layer]);
        free(net->bias[layer]);
    }

    #if NN_DEBUG_PRINT
        total_freed += malloc_size(net->weights);
        total_freed += malloc_size(net->bias);
        printf("freed network weights and biases: %.6f MiB\n", (float)total_freed / (1024 * 1024));
    #endif
    free(net->weights);
    free(net->bias);

    for (unsigned int i = 0; i < net->layers; i++) {
        free(net->out[i]);
    }
    free(net->in);
    free(net->out);

    free(net->neurons_per_layer);
    free(net);

    #if NN_MEMORY_TRIM_AFTER_FREE
    #ifdef __linux__
        malloc_trim(0);
    #endif
    #endif
}


NN_trainer* NN_trainer_init(NN_network* network, NN_learning_settings* learning_settings, NN_use_settings* use_settings, char *device_name) {
    unsigned int _size = strlen(device_name) + 1;

    NN_trainer* trainer = (NN_trainer*)malloc(sizeof(NN_trainer));
    trainer->processor.device_name = malloc(_size);
    for (unsigned int i = 0; i < _size; i++) {
        trainer->processor.device_name[i] = device_name[i];
    }

    trainer->learning_settings = learning_settings;
    trainer->processor.network = network;
    trainer->processor.settings = use_settings;

    // malloc weight+bias gradients if using batching
    if (learning_settings->use_batching) {
        unsigned int layers = network->layers;
        unsigned int* neurons_per_layer = network->neurons_per_layer;

        trainer->grad_weights = malloc(sizeof(float**) * (layers-1));
        trainer->grad_bias = malloc(sizeof(float*) * (layers-1));

        for (unsigned int layer = 0; layer < layers - 1; layer++) {
            unsigned int in_n = neurons_per_layer[layer];
            unsigned int out_n = neurons_per_layer[layer + 1];
    
            trainer->grad_weights[layer] = malloc(sizeof(float*) * in_n);
            trainer->grad_bias[layer] = calloc(out_n, sizeof(float));
    
            for (unsigned int i = 0; i < in_n; i++) {
                trainer->grad_weights[layer][i] = calloc(out_n, sizeof(float));
            }
        }
    }

    return trainer;
}

void NN_trainer_free(NN_trainer* trainer) {

    // first free gradient buffers if using batching
    if (trainer->learning_settings->use_batching) {
        unsigned int layers = trainer->processor.network->layers;
        unsigned int* neurons_per_layer = trainer->processor.network->neurons_per_layer;
        
        #if NN_DEBUG_PRINT
            size_t total_freed = 0;
        #endif

        for (unsigned int layer = 0; layer < layers - 1; layer++) {
            unsigned int in_n = neurons_per_layer[layer];
    
            for (unsigned int i = 0; i < in_n; i++) {
                #if NN_DEBUG_PRINT
                    total_freed += malloc_size(trainer->grad_weights[layer][i]);
                #endif
                free(trainer->grad_weights[layer][i]);
            }
            #if NN_DEBUG_PRINT
                total_freed += malloc_size(trainer->grad_weights[layer]);
                total_freed += malloc_size(trainer->grad_bias[layer]);
            #endif
            free(trainer->grad_weights[layer]);
            free(trainer->grad_bias[layer]);
        }
    
        #if NN_DEBUG_PRINT
            total_freed += malloc_size(trainer->grad_weights);
            total_freed += malloc_size(trainer->grad_bias);
            printf("freed training gradient buffers: %.6f MiB\n", (float)total_freed / (1024 * 1024));
        #endif
        free(trainer->grad_weights);
        free(trainer->grad_bias);
    }

    free(trainer);
    free(trainer->processor.device_name);
}

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
    unsigned int layers = net->layers;

    for (unsigned int i = 0; i < net->neurons_per_layer[0]; i++) {
        net->in[i] = in[i];
        net->out[0][i] = in[i];
    }

    // forward pass
    for (unsigned int layer = 1; layer < layers; layer++) {
        for (unsigned int j = 0; j < net->neurons_per_layer[layer]; j++) {
            float sum = net->bias[layer - 1][j];
            for (unsigned int i = 0; i < net->neurons_per_layer[layer - 1]; i++) {
                sum += net->out[layer - 1][i] * net->weights[layer - 1][i][j];
            }

            // apply activation
            switch (processor->settings->activation) {
                case RELU:
                    net->out[layer][j] = sum > 0 ? sum : 0;
                    break;
                case SIGMOID:
                    net->out[layer][j] = 1.0f / (1.0f + expf(-sum));
                    break;
                case TANH:
                    net->out[layer][j] = tanhf(sum);
                    break;
                case SOFTMAX:
                    // handled at output layer
                    net->out[layer][j] = sum; 
                    break;
            }
        }

        if (processor->settings->activation == SOFTMAX && layer == layers - 1) {
            float max = net->out[layer][0];
            for (unsigned int i = 1; i < net->neurons_per_layer[layer]; i++)
                if (net->out[layer][i] > max) max = net->out[layer][i];

            float sum = 0.0f;
            for (unsigned int i = 0; i < net->neurons_per_layer[layer]; i++) {
                net->out[layer][i] = expf(net->out[layer][i] - max);
                sum += net->out[layer][i];
            }
            for (unsigned int i = 0; i < net->neurons_per_layer[layer]; i++) {
                net->out[layer][i] /= sum;
            }
        }
    }

    unsigned int last = layers - 1;
    for (unsigned int i = 0; i < net->neurons_per_layer[last]; i++) {
        out[i] = net->out[last][i];
    }
}

void NN_trainer_train(NN_trainer* trainer, float* in, float* desired_out) {
    NN_network* net = trainer->processor.network;
    NN_use_settings* settings = trainer->processor.settings;
    float lr = trainer->learning_settings->learning_rate;
    unsigned int layers = net->layers;

    // run forward pass
    NN_processor_process(&trainer->processor, in, net->out[layers - 1]);

    // allocate deltas
    float** deltas = malloc(sizeof(float*) * layers);
    for (unsigned int l = 0; l < layers; l++) {
        deltas[l] = calloc(net->neurons_per_layer[l], sizeof(float));
    }

    // compute output layer error
    unsigned int out_layer = layers - 1;
    for (unsigned int i = 0; i < net->neurons_per_layer[out_layer]; i++) {
        float out = net->out[out_layer][i];
        float err = out - desired_out[i];
        float deriv = 1.0f;
        switch (settings->activation) {
            case SIGMOID:
                deriv = out * (1.0f - out);
                break;
            case TANH:
                deriv = 1.0f - out * out;
                break;
            case RELU:
                deriv = (out > 0) ? 1.0f : 0.0f;
                break;
            case SOFTMAX:
                deriv = 1.0f; // usually combined with cross-entropy loss
                break;
        }
        deltas[out_layer][i] = err * deriv;
    }

    // backpropagation for hidden layers
    for (int l = layers - 2; l >= 0; l--) {
        for (unsigned int i = 0; i < net->neurons_per_layer[l]; i++) {
            float out = net->out[l][i];
            float delta_sum = 0.0f;
            for (unsigned int j = 0; j < net->neurons_per_layer[l + 1]; j++) {
                delta_sum += net->weights[l][i][j] * deltas[l + 1][j];
            }
            float deriv = 1.0f;
            switch (settings->activation) {
                case SIGMOID: deriv = out * (1.0f - out); break;
                case TANH: deriv = 1.0f - out * out; break;
                case RELU: deriv = (out > 0) ? 1.0f : 0.0f; break;
                default: break;
            }
            deltas[l][i] = delta_sum * deriv;
        }
    }

    // update weights and biases
    for (unsigned int l = 0; l < layers - 1; l++) {
        for (unsigned int i = 0; i < net->neurons_per_layer[l]; i++) {
            for (unsigned int j = 0; j < net->neurons_per_layer[l + 1]; j++) {
                net->weights[l][i][j] -= lr * deltas[l + 1][j] * net->out[l][i];
            }
        }
        for (unsigned int j = 0; j < net->neurons_per_layer[l + 1]; j++) {
            net->bias[l][j] -= lr * deltas[l + 1][j];
        }
    }

    // free deltas
    for (unsigned int l = 0; l < layers; l++) {
        free(deltas[l]);
    }
    free(deltas);
}

void NN_trainer_accumulate(NN_trainer *trainer, float *input, float *target) {
    if (!trainer->learning_settings->use_batching) { fprintf(stderr, "ERROR: can't accumulate to gradient buffers because they have not been allocated\n    FIX: enable use_batching in learning_settings, this will roughly double memory usage though as a result");}
    NN_network* net = trainer->processor.network;
    NN_use_settings* settings = trainer->processor.settings;
    float lr = trainer->learning_settings->learning_rate;
    unsigned int layers = net->layers;

    // run forward pass
    NN_processor_process(&trainer->processor, input, net->out[layers - 1]);

    // allocate deltas
    float** deltas = malloc(sizeof(float*) * layers);
    for (unsigned int l = 0; l < layers; l++) {
        deltas[l] = calloc(net->neurons_per_layer[l], sizeof(float));
    }

    // compute output layer error
    unsigned int out_layer = layers - 1;
    for (unsigned int i = 0; i < net->neurons_per_layer[out_layer]; i++) {
        float out = net->out[out_layer][i];
        float err = out - target[i];
        float deriv = 1.0f;
        switch (settings->activation) {
            case SIGMOID:
                deriv = out * (1.0f - out);
                break;
            case TANH:
                deriv = 1.0f - out * out;
                break;
            case RELU:
                deriv = (out > 0) ? 1.0f : 0.0f;
                break;
            case SOFTMAX:
                deriv = 1.0f; // usually combined with cross-entropy loss
                break;
        }
        deltas[out_layer][i] = err * deriv;
    }

    // backpropagation for hidden layers
    for (int l = layers - 2; l >= 0; l--) {
        for (unsigned int i = 0; i < net->neurons_per_layer[l]; i++) {
            float out = net->out[l][i];
            float delta_sum = 0.0f;
            for (unsigned int j = 0; j < net->neurons_per_layer[l + 1]; j++) {
                delta_sum += net->weights[l][i][j] * deltas[l + 1][j];
            }
            float deriv = 1.0f;
            switch (settings->activation) {
                case SIGMOID: deriv = out * (1.0f - out); break;
                case TANH: deriv = 1.0f - out * out; break;
                case RELU: deriv = (out > 0) ? 1.0f : 0.0f; break;
                default: break;
            }
            deltas[l][i] = delta_sum * deriv;
        }
    }

    // add updates to gradient buffer
    for (unsigned int l = 0; l < layers - 1; l++) {
        for (unsigned int i = 0; i < net->neurons_per_layer[l]; i++) {
            for (unsigned int j = 0; j < net->neurons_per_layer[l + 1]; j++) {
                trainer->grad_weights[l][i][j] += deltas[l + 1][j] * net->out[l][i];
            }
        }
        for (unsigned int j = 0; j < net->neurons_per_layer[l + 1]; j++) {
            trainer->grad_bias[l][j] += deltas[l + 1][j];
        }
    }

    // free deltas
    for (unsigned int l = 0; l < layers; l++) {
        free(deltas[l]);
    }
    free(deltas);
}

void NN_trainer_apply(NN_trainer *trainer, unsigned int batch_size) {
    //update weights and biases
    for (unsigned int l = 0; l < trainer->processor.network->layers - 1; l++) {
        for (unsigned int i = 0; i < trainer->processor.network->neurons_per_layer[l]; i++) {
            for (unsigned int j = 0; j < trainer->processor.network->neurons_per_layer[l + 1]; j++) {
                trainer->processor.network->weights[l][i][j] -= trainer->learning_settings->learning_rate * (trainer->grad_weights[l][i][j] / batch_size);
                trainer->grad_weights[l][i][j] = 0;
            }
        }
        for (unsigned int j = 0; j < trainer->processor.network->neurons_per_layer[l + 1]; j++) {
            trainer->processor.network->bias[l][j] -= trainer->learning_settings->learning_rate * (trainer->grad_bias[l][j] / batch_size);
            trainer->grad_bias[l][j] = 0;
        }
    }
}

float NN_trainer_loss(NN_trainer* trainer, float* desired) {
    NN_network* net = trainer->processor.network;
    float loss = 0.0f;
    unsigned int last = net->layers - 1;
    unsigned int out_n = net->neurons_per_layer[last];
    for (unsigned int i = 0; i < out_n; i++) {
        float diff = net->out[last][i] - desired[i];
        loss += diff * diff;
    }
    return loss / out_n;
}

/* OLD CODE


NN_layer NN_layer_init(float* weights, float* bias, unsigned int size) {
    return (NN_layer){.weights = weights, .bias = bias, .size = size};
}
void NN_layer_destroy(NN_layer* layer) {
    free(layer->weights);
    free(layer->bias);
    free(layer);
}

void NN_layer_process(float *input, NN_layer *layer1, NN_layer *layer2, float *output) {
    printf("layer...\n");
    for (unsigned int neuron = 0; neuron < layer1->size; neuron++) {
        printf("neuron(%i)\n",neuron);
        printf("    -bias: %f\n",layer1->bias[neuron]);
        printf("    -weights:\n");
        for (unsigned int weight = 0; weight < layer2->size; weight++) {
            printf("        -%f\n",layer1->weights[neuron*layer1->size + weight]);
            output[weight] += (input[neuron] * layer1->weights[neuron*layer1->size + weight]) + layer1->bias[neuron];
        }
    }
}


NN_network NN_network_init(NN_layer* layers, unsigned int n_layers, unsigned int n_neurons) {
    return (NN_network){.layers = layers, .n_layers = n_layers, .n_neurons = n_neurons};
}
void NN_network_destroy(NN_network* network) {
    for (unsigned int layer = 0; layer < network->n_layers; layer++) {
        NN_layer_destroy(&network->layers[layer]);
    }
    free(network->layers);
    free(network);
}


float* NN_network_process(NN_network* network, float* output, float* input) {
    for (unsigned int layer = 0; layer < network->n_layers-1; layer++) {
        memset(output, 0, sizeof(float) * network->n_neurons);
        NN_layer_process(input, &network->layers[layer], &network->layers[layer+1], output);
        float *tmp = input;
        input = output;
        output = tmp;
    }
    float *tmp = input;
    input = output;
    output = tmp;
    return output;
}

float* NN_network_forward(NN_network* network, float* output, float* input) {
    float* intermediates = malloc(sizeof(float) * network->n_layers * network->n_neurons);
    float* input_copy = malloc(sizeof(float) * network->n_neurons);
    for (int i = 0; i < network->n_neurons; i++) {
        input_copy[i] = input[i];
    }
    for (unsigned int layer = 0; layer < network->n_layers-1; layer++) {
        memset(output, 0, sizeof(float) * network->n_neurons);
        NN_layer_process(input_copy, &network->layers[layer], &network->layers[layer+1], output);

        // copy output into intermediate
        for (unsigned int i = 0; i < network->n_neurons; i++) {
            intermediates[layer*network->n_layers + i] = output[i];
        }

        float *tmp = input_copy;
        input_copy = output;
        output = tmp;
    }
    float *tmp = input_copy;
    input_copy = output;
    output = tmp;
    free(input_copy);
    return intermediates;
}

void NN_network_backprop(NN_network *network, float *real_output, float *desired_output, float* intermediates, float learning_rate) {
    float* deltas = malloc(sizeof(float) * network->n_neurons * network->n_layers);

    // output layer error
    for (unsigned int real_data_idx = 0; real_data_idx < network->n_neurons; real_data_idx++) {
        float error = desired_output[real_data_idx] - desired_output[real_data_idx];
        float error_delta = error * real_output[real_data_idx] * (1-real_output[real_data_idx]);
        deltas[(network->n_layers-1)*network->n_layers +real_data_idx] = error_delta;
    }

    // hidden layer error
    for (unsigned int layer = network->n_layers-2; layer > 0; layer--) {
        for (unsigned int h_neuron = 0; h_neuron < network->n_neurons; h_neuron++) {
            float h_error = 0;
            for (unsigned int h_conn = 0; h_conn < network->n_neurons; h_conn++) {
                h_error += network->layers[layer+1].weights[h_neuron*network->layers[layer+1].size + h_conn] * deltas[(layer+1) * network->n_layers + h_conn];
            }
            deltas[layer*network->n_layers + h_neuron] = h_error * deltas[(layer+1) * network->n_layers + h_neuron] * (1-deltas[(layer+1) * network->n_layers + h_neuron]);
        }
    }

    printf("updating network...\n");
    // update network
    for (unsigned int layer = 0; layer < network->n_layers-1; layer++) {
        for (unsigned int neuron = 0; neuron < network->n_neurons; neuron++) {
            // update weights
            for (unsigned int weight = 0; weight < network->n_neurons; weight++) {
                float delta_weight = deltas[(layer+1) * network->n_layers + weight] * intermediates[layer*network->n_layers + neuron];
                network->layers[layer].weights[neuron*network->layers[layer].size + weight] -= learning_rate * delta_weight;
            }

            // update bias
            float delta_bias = deltas[(layer+1) * network->n_layers + neuron];
            network->layers[layer].bias[neuron] -= learning_rate * delta_bias;
        }
    }

}

float NN_network_loss(NN_network* network, float *real_output, float *desired_output) {
    float loss = 0;
    for (unsigned int neuron = 0; neuron < network->n_neurons; neuron++) {
        loss += (0.5)*(real_output[neuron]-desired_output[neuron])*(real_output[neuron]-desired_output[neuron]);
    }
    return loss;
}
*/