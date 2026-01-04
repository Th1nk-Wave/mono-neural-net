#include "layers/recurrent.h"
#include "NN.h"
#include "RNG.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <wchar.h>

void NN_rnn_forward(NN_layer* layer, const float* input) {
    NN_layer_rnn_params* p = (NN_layer_rnn_params*)layer->params;

    for (unsigned int o = 0; o < layer->out_size; o++) {
        float sum = p->bias[o];

        // input contribution
        for (unsigned int i = 0; i < layer->in_size; i++)
            sum += p->weights_input[o][i] * input[i];

        // hidden contribution
        for (unsigned int h = 0; h < layer->out_size; h++)
            sum += p->weights_hidden[o][h] * p->hidden_state[h];

        sum = NN_apply_activation(layer->activation, sum);
        layer->out[o] = sum;
    }

    // update hidden state
    memcpy(p->hidden_state, layer->out, sizeof(float) * layer->out_size);
}

void NN_rnn_randomise(NN_layer* layer, float min, float max) {
    NN_layer_rnn_params* p = (NN_layer_rnn_params*)layer->params;
    float limit = sqrtf(6.0f / (layer->in_size + layer->out_size));

    for (unsigned int o = 0; o < layer->out_size; o++) {
        for (unsigned int i = 0; i < layer->in_size; i++)
            p->weights_input[o][i] = random_float_range(-limit, limit);

        for (unsigned int h = 0; h < layer->out_size; h++)
            p->weights_hidden[o][h] = random_float_range(-limit, limit);

        p->bias[o] = 0.0f;
        p->hidden_state[o] = 0.0f;
    }
}

void NN_rnn_backward(NN_training_layer* layer, const float* input, const float* delta_next, float* delta_out) {
    NN_layer* base = layer->base;
    NN_layer_rnn_params* p = (NN_layer_rnn_params*)base->params;
    NN_layer_rnn_grads_buf* g = (NN_layer_rnn_grads_buf*)layer->grads;

    memset(delta_out, 0, sizeof(float) * base->in_size);

    for (unsigned int o = 0; o < base->out_size; o++) {
        float d_act = NN_activation_deriv(base->activation, base->out[o]);
        float delta = delta_next[o] * d_act;

        g->grad_bias[o] += delta;

        for (unsigned int i = 0; i < base->in_size; i++) {
            g->grad_weights_input[o][i] += delta * input[i];
            delta_out[i] += p->weights_input[o][i] * delta;
        }

        for (unsigned int h = 0; h < base->out_size; h++) {
            g->grad_weights_hidden[o][h] += delta * p->hidden_state[h];
            g->grad_hidden_state[h] += p->weights_hidden[o][h] * delta;
        }
    }
}

void NN_rnn_apply(NN_training_layer* layer, unsigned int batch_size) {
    NN_learning_settings* ls = layer->settings;
    float lr = ls->learning_rate;

    NN_layer* base = layer->base;
    NN_layer_rnn_params* p = (NN_layer_rnn_params*)base->params;
    NN_layer_rnn_grads_buf* g = (NN_layer_rnn_grads_buf*)layer->grads;

    NN_ADAM_buf* adam = (NN_ADAM_buf*)layer->optimizer_buf;

    adam->adam_t++;

    // weights_input
    for (unsigned int o = 0; o < base->out_size; o++) {
        for (unsigned int i = 0; i < base->in_size; i++) {
            float grad = g->grad_weights_input[o][i] / batch_size;

            float m = adam->m_weights[o][i];
            float v = adam->v_weights[o][i];

            m = 0.9f * m + 0.1f * grad;
            v = 0.999f * v + 0.001f * (grad * grad);

            adam->m_weights[o][i] = m;
            adam->v_weights[o][i] = v;

            float m_hat = m / (1.0f - powf(0.9f, (float)adam->adam_t));
            float v_hat = v / (1.0f - powf(0.999f, (float)adam->adam_t));

            p->weights_input[o][i] -= lr * (m_hat / (sqrtf(v_hat) + 1e-8f));
            g->grad_weights_input[o][i] = 0.0f;
            //printf("rnn input weight [%i][%i]: %f\n",o,i,p->weights_input[o][i]);

        }
    }

    // weights_hidden
    for (unsigned int o = 0; o < base->out_size; o++) {
        for (unsigned int h = 0; h < base->out_size; h++) {
            float grad = g->grad_weights_hidden[o][h] / batch_size;

            float m = adam->m_weights[o][h];
            float v = adam->v_weights[o][h];

            m = 0.9f * m + 0.1f * grad;
            v = 0.999f * v + 0.001f * (grad * grad);

            adam->m_weights[o][h] = m;
            adam->v_weights[o][h] = v;

            float m_hat = m / (1.0f - powf(0.9f, (float)adam->adam_t));
            float v_hat = v / (1.0f - powf(0.999f, (float)adam->adam_t));
            //printf("v_hat: %f\n",v_hat);

            p->weights_hidden[o][h] -= lr * (m_hat / (sqrtf(v_hat) + 1e-8f));
            //printf("rnn weight [%i][%i]: %f\n",o,h,p->weights_hidden[o][h]);
            g->grad_weights_hidden[o][h] = 0.0f;
        }
    }

    for (unsigned int o = 0; o < base->out_size; o++) {
        p->bias[o] -= lr * g->grad_bias[o] / batch_size;
        //printf("bias [%f]\n",p->bias[o]);
        g->grad_bias[o] = 0.0f;
        g->grad_hidden_state[o] = 0.0f;
    }
}

// creation
NN_layer* NN_create_rnn_layer(unsigned int n_in, unsigned int n_out, NN_activation_function activation) {
    NN_layer* layer = NN_layer_init(n_in, n_out, activation);

    NN_layer_rnn_params* params = (NN_layer_rnn_params*)malloc(sizeof(NN_layer_rnn_params));
    params->bias = calloc(n_out, sizeof(float));
    params->hidden_state = calloc(n_out, sizeof(float));

    params->weights_input = malloc(sizeof(float*) * n_out);
    params->weights_hidden = malloc(sizeof(float*) * n_out);
    for (unsigned int i = 0; i < n_out; i++) {
        params->weights_input[i] = malloc(sizeof(float) * n_in);
        params->weights_hidden[i] = malloc(sizeof(float) * n_out);
    }

    layer->params = params;
    layer->type = RECURRENT;
    layer->forward = NN_rnn_forward;
    layer->randomize = NN_rnn_randomise;
    return layer;
}

void NN_clean_up_rnn_layer(NN_layer* layer) {
    NN_layer_rnn_params* p = (NN_layer_rnn_params*)layer->params;
    for (unsigned int i = 0; i < layer->out_size; i++) {
        free(p->weights_input[i]);
        free(p->weights_hidden[i]);
    }
    free(p->weights_input);
    free(p->weights_hidden);
    free(p->bias);
    free(p->hidden_state);
    free(p);
}

void NN_set_rnn_training_layer(NN_training_layer* layer, NN_learning_settings* settings) {
    if (settings->use_batching) {
        NN_layer_rnn_grads_buf* g = (NN_layer_rnn_grads_buf*)malloc(sizeof(NN_layer_rnn_grads_buf));
        g->grad_bias = calloc(layer->base->out_size, sizeof(float));
        g->grad_hidden_state = calloc(layer->base->out_size, sizeof(float));

        g->grad_weights_input = malloc(sizeof(float*) * layer->base->out_size);
        g->grad_weights_hidden = malloc(sizeof(float*) * layer->base->out_size);
        for (unsigned int i = 0; i < layer->base->out_size; i++) {
            g->grad_weights_input[i] = calloc(layer->base->in_size, sizeof(float));
            g->grad_weights_hidden[i] = calloc(layer->base->out_size, sizeof(float));
        }

        layer->grads = g;
    }

    layer->backward = NN_rnn_backward;
    layer->apply = NN_rnn_apply;
}

void NN_clean_up_rnn_training_layer(NN_training_layer* layer) {
    if (layer->settings->use_batching) {
        NN_layer_rnn_grads_buf* g = (NN_layer_rnn_grads_buf*)layer->grads;

        free(g->grad_bias);
        free(g->grad_hidden_state);

        for (unsigned int i = 0; i < layer->base->out_size; i++) {
            free(g->grad_weights_input[i]);
            free(g->grad_weights_hidden[i]);
        }

        free(g->grad_weights_input);
        free(g->grad_weights_hidden);
        free(g);
    }
}


int NN_rnn_save_to_file(NN_layer *layer, FILE *f) {

    // metadata
    NN_layer_type layer_type = RECURRENT;
    if (fwrite(&layer_type, sizeof(NN_layer_type), 1, f) != 1) goto error;
    if (fwrite(&layer->activation, sizeof(NN_activation_function), 1, f) != 1) goto error;
    if (fwrite(&layer->in_size, sizeof(uint32_t), 1, f) != 1) goto error;
    if (fwrite(&layer->out_size, sizeof(uint32_t), 1, f) != 1) goto error;

    NN_layer_rnn_params* params = layer->params;

    // weights input + hidden
    for (unsigned int w = 0; w < layer->out_size; w++) {
        if (fwrite(params->weights_input[w], sizeof(float), layer->in_size, f) != layer->in_size) goto error;
        if (fwrite(params->weights_hidden[w], sizeof(float), layer->out_size, f) != layer->out_size) goto error;
    }

    // bias
    if (fwrite(params->bias, sizeof(float), layer->out_size, f) != layer->out_size) goto error;

    // hidden state
    if (fwrite(params->hidden_state, sizeof(float), layer->out_size, f) != layer->out_size) goto error;
    
    return 1;


error:
    printf("[ERROR] error in NN_rnn_save_to_file\n");
    return -1;
}

NN_layer* NN_rnn_init_from_file(FILE* f) {

    // metadata
    NN_activation_function activation;
    uint32_t in_size;
    uint32_t out_size;

    if (fread(&activation, sizeof(NN_activation_function), 1, f) != 1) goto error;
    if (fread(&in_size,sizeof(uint32_t), 1, f) != 1) goto error;
    if (fread(&out_size, sizeof(uint32_t), 1, f) != 1) goto error;

    // create
    NN_layer* layer = NN_create_rnn_layer(in_size, out_size, activation);
    NN_layer_rnn_params* params = layer->params;

    // weights input + hidden
    for (unsigned int w = 0; w < layer->out_size; w++) {
        if (fread(params->weights_input[w], sizeof(float), layer->in_size, f) != layer->in_size) goto error_fill;
        if (fread(params->weights_hidden[w], sizeof(float), layer->out_size, f) != layer->out_size) goto error_fill;
    }

    // bias
    if (fread(params->bias, sizeof(float), layer->out_size, f) != layer->out_size) goto error_fill;

    // hidden state
    if (fread(params->hidden_state, sizeof(float), layer->out_size, f) != layer->out_size) goto error_fill;

    return layer;


error_fill:
    NN_layer_free(layer);

error:
    printf("[ERROR] error in NN_rnn_init_from_file\n");
    return NULL;
}