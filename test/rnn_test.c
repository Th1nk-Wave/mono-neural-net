#include "layers/fully_connected.h"
#include "layers/recurrent.h"
#include "NN.h"


#include <math.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>

#define RANDOM_INIT_MAX 0.1
#define RANDOM_INIT_MIN -0.1

#define layers_N 4
#define in_param 1
#define hidden_param 1
#define out_param 1

#define TBPTT_STEPS 4
#define SEQ_LEN 10

void main() {
    NN_use_settings* use_settings = (NN_use_settings*)malloc(sizeof(NN_use_settings));
    use_settings->device_type = AUTO;

    NN_learning_settings* learning_settings = (NN_learning_settings*)malloc(sizeof(NN_learning_settings));
    learning_settings->learning_rate = 0.02;
    learning_settings->loss_function = HUBER;
    learning_settings->use_batching = true;
    learning_settings->optimizer = ADAM;
    learning_settings->use_tbptt = 2;

    NN_network* network = NN_network_init_from_file("sample_net.net",true);

    if (!network) {
        NN_layer** layers = (NN_layer**)malloc(sizeof(NN_layer*)*layers_N);
    
        layers[0] = NN_create_fully_connected_layer(in_param, hidden_param, TANH);
        //layers[1] = NN_create_fully_connected_layer(hidden_param, hidden_param, LERELU);
        //layers[2] = NN_create_fully_connected_layer(hidden_param, hidden_param, LERELU);
        layers[3] = NN_create_fully_connected_layer(hidden_param, out_param, TANH);

    
        //layers[0] = NN_create_rnn_layer(in_param, hidden_param, TANH);
        layers[1] = NN_create_rnn_tbptt_layer(hidden_param, hidden_param, SIGMOID,2);
        layers[2] = NN_create_rnn_tbptt_layer(hidden_param, hidden_param, SIGMOID,2);
        //layers[3] = NN_create_rnn_layer(hidden_param, out_param, SIGMOID);


        network = NN_network_init(layers,layers_N);
        NN_network_randomise_xaivier(network, RANDOM_INIT_MIN, RANDOM_INIT_MAX);
    }

    NN_processor* processor = NN_processor_init(network, use_settings, "AUTO");
    NN_trainer* trainer = NN_trainer_init(processor, learning_settings);


    // train
    unsigned int batch_size = 10;
    

    float x[SEQ_LEN] = {
    0.1, 0.2, 0.4, 0.8, 0.14,
    0.28, 0.56, 0.14, 0.3, 0.9
    };

    float y[SEQ_LEN];
    y[0] = 0.0f; // no previous context
    for (int i = 1; i < SEQ_LEN; i++)
        y[i] = x[i] + x[i-1];

    unsigned int epoch = 0;
    float loss = 0;
    for (unsigned int epoch = 0; epoch < 30000; epoch++) {
        loss = 0.0f;

        
        for (unsigned int start = 1; start + TBPTT_STEPS <= SEQ_LEN; start++) {

            // reset RNN state between windows
            NN_network_reset_state(network);
            for (unsigned int t = 0; t < TBPTT_STEPS; t++) {
                unsigned int idx = start + t;

                NN_trainer_accumulate(
                    trainer,
                    &x[idx],   // input x_t
                    &y[idx]    // target y_t
                );

                loss += NN_trainer_loss(trainer, &y[idx]);
            }

            NN_trainer_apply(trainer, TBPTT_STEPS);
        }

        if (epoch % 20 == 0) {
            printf("epoch %u, loss: %f\n",
               epoch, loss / (SEQ_LEN - 1));
        }
    }

    float out[2] = {0,0};

    printf("\nTesting sequence prediction\n");

    NN_network_reset_state(network);

    for (unsigned int i = 1; i < SEQ_LEN; i++) {
        float out = 0.0f;

        NN_processor_process(processor, &x[i], &out);

        printf(
            "t=%u  x=%f  expected=%f  got=%f\n",
            i, x[i], y[i], out
        );
    }


    // save to file
    printf("saving network...\n");
    NN_network_save_to_file(network, "sample_net.net");

    // clean up
    NN_trainer_free(trainer);
    NN_processor_free(processor);
    NN_network_free(network);

    free(learning_settings);
    free(use_settings);
}

