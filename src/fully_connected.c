#include "layers/fully_connected.h"
#include "NN.h"
#include "RNG.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void NN_fully_connected_forward(NN_layer* layer, const float* input) {
    NN_layer_fully_connected_params* p = (NN_layer_fully_connected_params*)layer->params;

    for (unsigned int o = 0; o < layer->out_size; o++) {
        float sum = p->bias[o];

        for (unsigned int i = 0; i < layer->in_size; i++)
            sum += p->weights[o][i] * input[i];

        // activation
        sum = NN_apply_activation(layer->activation, sum);

        layer->out[o] = sum;
    }
}

void NN_fully_connected_randomise(NN_layer *layer, float min, float max) {
    NN_layer_fully_connected_params* p = (NN_layer_fully_connected_params*)layer->params;
    unsigned int in_n = layer->in_size;
    unsigned int out_n = layer->out_size;
    float limit = sqrtf(6.0f / (float)(in_n + out_n));
    for (unsigned int i = 0; i < out_n; i++) {
        for (unsigned int j = 0; j < in_n; j++) {
            p->weights[i][j] = random_float_range(-limit, limit);
        }
    }
    for (unsigned int j = 0; j < out_n; j++) {
        p->bias[j] = 0.0f;
    }
}

void NN_fully_connected_backward(NN_training_layer *layer, const float *input, const float *delta_next, float *delta_out) {
    NN_layer* base = layer->base;
    NN_layer_fully_connected_params* p = (NN_layer_fully_connected_params*)base->params;
    NN_layer_fully_connected_grads_buf* g = (NN_layer_fully_connected_grads_buf*)layer->grads;

    // zero delta_out
    memset(delta_out, 0, sizeof(float) * base->in_size);

    for (unsigned int o = 0; o < base->out_size; o++) {
        float d_act = NN_activation_deriv(base->activation, base->out[o]);

        float delta = delta_next[o] * d_act;

        g->grad_bias[o] += delta;

        for (unsigned int i = 0; i < base->in_size; i++) {
            g->grad_weights[o][i] += delta * input[i];
            delta_out[i] += p->weights[o][i] * delta;
        }
    }
}

void NN_fully_connected_apply(NN_training_layer *layer, unsigned int batch_size) {
    NN_learning_settings* ls = layer->settings;
    float lr = ls->learning_rate;

    NN_layer* base = layer->base;
    NN_layer_fully_connected_params* p = (NN_layer_fully_connected_params*)base->params;
    NN_layer_fully_connected_grads_buf* g = (NN_layer_fully_connected_grads_buf*)layer->grads;

    // ADAM / ADAMW
    if (ls->optimizer == ADAM || ls->optimizer == ADAMW) {
        NN_ADAM_buf* adam = (NN_ADAM_buf*)layer->optimizer_buf;

        const float beta1 = (ls->adam_beta1 > 0.0f) ? ls->adam_beta1 : 0.9f;
        const float beta2 = (ls->adam_beta2 > 0.0f) ? ls->adam_beta2 : 0.999f;
        const float eps   = (ls->adam_epsilon > 0.0f) ? ls->adam_epsilon : 1e-8f;
        const float wd    = (ls->weight_decay > 0.0f) ? ls->weight_decay : 0.01f;

        adam->adam_t++;

        float one_minus_beta1_t = 1.0f - powf(beta1, (float)adam->adam_t);
        float one_minus_beta2_t = 1.0f - powf(beta2, (float)adam->adam_t);

        // weights
        for (unsigned int o = 0; o < base->out_size; o++) {
            for (unsigned int i = 0; i < base->in_size; i++) {
                float grad = g->grad_weights[o][i] / (float)batch_size;

                float m = adam->m_weights[o][i];
                float v = adam->v_weights[o][i];

                m = beta1 * m + (1.0f - beta1) * grad;
                v = beta2 * v + (1.0f - beta2) * (grad * grad);

                adam->m_weights[o][i] = m;
                adam->v_weights[o][i] = v;

                float m_hat = m / one_minus_beta1_t;
                float v_hat = v / one_minus_beta2_t;

                if (ls->optimizer == ADAMW) {
                    p->weights[o][i] -= lr * wd * p->weights[o][i];
                }

                p->weights[o][i] -= lr * (m_hat / (sqrtf(v_hat) + eps));
                g->grad_weights[o][i] = 0.0f;
            }
        }

        // bias
        for (unsigned int o = 0; o < base->out_size; o++) {
            float grad = g->grad_bias[o] / (float)batch_size;

            float m = adam->m_bias[o];
            float v = adam->v_bias[o];

            m = beta1 * m + (1.0f - beta1) * grad;
            v = beta2 * v + (1.0f - beta2) * (grad * grad);

            adam->m_bias[o] = m;
            adam->v_bias[o] = v;

            float m_hat = m / one_minus_beta1_t;
            float v_hat = v / one_minus_beta2_t;

            if (ls->optimizer == ADAMW) {
                p->bias[o] -= lr * wd * p->bias[o];
            }

            p->bias[o] -= lr * (m_hat / (sqrtf(v_hat) + eps));
            g->grad_bias[o] = 0.0f;
        }

        return;
    }

    // fallback for gradient decent
    for (unsigned int o = 0; o < base->out_size; o++) {
        p->bias[o] -= lr * g->grad_bias[o] / batch_size;
        g->grad_bias[o] = 0.0f;

        for (unsigned int i = 0; i < base->in_size; i++) {
            p->weights[o][i] -= lr * g->grad_weights[o][i] / batch_size;
            g->grad_weights[o][i] = 0.0f;
        }
    }
}


// LAYER IMPLEMENTATION
NN_layer* NN_create_fully_connected_layer(unsigned int n_in, unsigned int n_out, NN_activation_function activation) {
    NN_layer* layer = NN_layer_init(n_in, n_out, activation);

    NN_layer_fully_connected_params* params = (NN_layer_fully_connected_params*)malloc(sizeof(NN_layer_fully_connected_params));
    params->bias = malloc(sizeof(float)*n_out);
    params->weights = malloc(sizeof(float*)*n_out);
    for (unsigned int i = 0; i < n_out; i++) {
        params->weights[i] = malloc(sizeof(float)*n_in);
    }
    layer->params = (void*)params;

    layer->type = FULLY_CONNECTED;
    layer->forward = NN_fully_connected_forward;
    layer->randomize = NN_fully_connected_randomise;
    return layer;
}
void NN_clean_up_fully_connected_layer(NN_layer *layer) {
    NN_layer_fully_connected_params* params = (NN_layer_fully_connected_params*)layer->params;
    for (unsigned int i = 0; i < layer->out_size; i++) {
        free(params->weights[i]);
    }
    free(params->weights);
    free(params->bias);
    free(params);
}


// TRAINING LAYER IMPLEMENTATION
void NN_set_fully_connected_training_layer(NN_training_layer* layer, NN_learning_settings* settings) {
    
    if (settings->use_batching) {
        NN_layer_fully_connected_grads_buf* grad_buff = (NN_layer_fully_connected_grads_buf*)malloc(sizeof(NN_layer_fully_connected_grads_buf));
        grad_buff->grad_bias = calloc(layer->base->out_size,sizeof(float));
        grad_buff->grad_weights = malloc(sizeof(float*)*layer->base->out_size);
        for (unsigned int i = 0; i < layer->base->out_size; i++) {
            grad_buff->grad_weights[i] = calloc(layer->base->in_size,sizeof(float));
        }
        layer->grads = grad_buff;
    }

    layer->backward = NN_fully_connected_backward;
    layer->apply = NN_fully_connected_apply;
}
void NN_clean_up_fully_connected_training_layer(NN_training_layer* layer) {
    if (layer->settings->use_batching) {
        NN_layer_fully_connected_grads_buf* grad_buff = (NN_layer_fully_connected_grads_buf*)layer->grads;
        free(grad_buff->grad_bias);
        for (unsigned int i = 0; i < layer->base->out_size; i++) {
            free(grad_buff->grad_weights[i]);
        }
        free(grad_buff->grad_weights);
        free(grad_buff);
    }
}


int NN_fully_connected_save_to_file(NN_layer *layer, FILE *f) {

    // metadata
    NN_layer_type layer_type = FULLY_CONNECTED;
    if (fwrite(&layer_type, sizeof(NN_layer_type), 1, f) != 1) goto error;
    if (fwrite(&layer->activation, sizeof(NN_activation_function), 1, f) != 1) goto error;
    if (fwrite(&layer->in_size, sizeof(uint32_t), 1, f) != 1) goto error;
    if (fwrite(&layer->out_size, sizeof(uint32_t), 1, f) != 1) goto error;

    NN_layer_fully_connected_params* params = layer->params;

    // weights
    for (unsigned int w = 0; w < layer->out_size; w++) {
        if(fwrite(params->weights[w], sizeof(float), layer->in_size, f) != layer->in_size) goto error;
    }

    // bias
    if (fwrite(params->bias, sizeof(float), layer->out_size, f) != layer->out_size) goto error;

    return 1;

error:
    printf("[ERROR] error in NN_fully_connected_save_to_file\n");
    return -1;
}

NN_layer* NN_fully_connected_init_from_file(FILE* f) {

    // metadata
    NN_activation_function activation;
    uint32_t in_size;
    uint32_t out_size;

    if (fread(&activation, sizeof(NN_activation_function), 1, f) != 1) goto error;
    if (fread(&in_size,sizeof(uint32_t), 1, f) != 1) goto error;
    if (fread(&out_size, sizeof(uint32_t), 1, f) != 1) goto error;

    // create
    NN_layer* layer = NN_create_fully_connected_layer(in_size, out_size, activation);
    NN_layer_fully_connected_params* params = layer->params;

    // weights
    for (unsigned int w = 0; w < layer->out_size; w++) {
        if(fread(params->weights[w], sizeof(float), layer->in_size, f) != layer->in_size) goto error_fill;
    }

    // bias
    if (fread(params->bias, sizeof(float), layer->out_size, f) != layer->out_size) goto error_fill;

    return layer;

error_fill:
    NN_layer_free(layer);

error:
    printf("[ERROR] error in NN_fully_connected_init_from_file\n");
    return NULL;
}