#include "layers/fully_connected.h"
#include "layers/recurrent.h"
#include "NN2.h"


#include <math.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>

#define RANDOM_INIT_MAX 0.1
#define RANDOM_INIT_MIN -0.1

#define layers_N 4
#define in_param 1
#define hidden_param 3
#define out_param 1

void shuffle(unsigned int *array, size_t n) {
    for (size_t i = n - 1; i > 0; i--) {
        size_t j = rand() % (i + 1);
        unsigned int tmp = array[i];
        array[i] = array[j];
        array[j] = tmp;
    }
}

void main() {
    NN_use_settings* use_settings = (NN_use_settings*)malloc(sizeof(NN_use_settings));
    use_settings->device_type = AUTO;

    NN_learning_settings* learning_settings = (NN_learning_settings*)malloc(sizeof(NN_learning_settings));
    learning_settings->learning_rate = 0.02;
    learning_settings->loss_function = HUBER;
    learning_settings->use_batching = true;
    learning_settings->optimizer = ADAM;

    NN_network* network = NN_network_init_from_file("sample_net.net");

    if (!network) {
        NN_layer** layers = (NN_layer**)malloc(sizeof(NN_layer*)*layers_N);
    
        layers[0] = NN_create_fully_connected_layer(in_param, hidden_param, TANH);
        //layers[1] = NN_create_fully_connected_layer(hidden_param, hidden_param, LERELU);
        //layers[2] = NN_create_fully_connected_layer(hidden_param, hidden_param, LERELU);
        layers[3] = NN_create_fully_connected_layer(hidden_param, out_param, TANH);
    
        //layers[0] = NN_create_rnn_layer(in_param, hidden_param, TANH);
        layers[1] = NN_create_rnn_layer(hidden_param, hidden_param, SIGMOID);
        layers[2] = NN_create_rnn_layer(hidden_param, hidden_param, SIGMOID);
        //layers[3] = NN_create_rnn_layer(hidden_param, out_param, SIGMOID);

        network = NN_network_init(layers,layers_N);
        NN_network_randomise_xaivier(network, RANDOM_INIT_MIN, RANDOM_INIT_MAX);
    }

    NN_processor* processor = NN_processor_init(network, use_settings, "AUTO");
    NN_trainer* trainer = NN_trainer_init(processor, learning_settings);

    // do training loop

    // train
    unsigned int batch_size = 10;
    /*
    float training_data[10][2][2] = {
        {{0,0},{0,0}},
        {{0,1},{0,1}},
        {{1,0},{1,0}},
        {{1,1},{1,1}},
        {{0.1,0.1},{0.1,0.1}},
        {{0,0.5},{0,0.5}},
        {{0.5,0},{0.5,0}},
        {{-0.9,-0.9},{-0.9,-0.9}},
        {{-1,-1},{-1,-1}},
        {{1,-1},{1,-1}},
    };
    */
    

    float training_data[10][2] = {
        {0.1,0.2},
        {0.2,0.4},
        {0.4,0.8},
        {0.8,0.14},
        {0.14,0.28},
        {0.28,0.56},
        {0.56,0.14},
        {0.14,0.3},
        {0.3,0.9},
        {0.9,0.14},
    };

    unsigned int idx[10] = {0,1,2,3,4,5,6,7,8,9};

    unsigned int epoch = 0;
    float loss = 0;
    for (unsigned int i = 0; i < 100; i++) {
        //shuffle(idx, 10);
        loss = 0;
        for (unsigned int batch = 0; batch < batch_size; batch++) {
            unsigned int index = idx[batch];
            NN_trainer_accumulate(trainer, &training_data[index][0],&training_data[index][1]);
            loss += NN_trainer_loss(trainer, &training_data[index][1]);
        }
        printf("epoch %i, loss: %f\n", epoch, loss/batch_size);
        NN_trainer_apply(trainer, batch_size);
        epoch++;
    }

    float out[2] = {0,0};

    printf("testing cases\n");
    for (unsigned int batch = 0; batch < batch_size; batch++) {
        unsigned int index = idx[batch];
        NN_processor_process(processor, &training_data[index][0],out);
        loss += NN_trainer_loss(trainer, &training_data[index][1]);
        printf("test case: [%u] expected {%f,%f} result {%f}\n",index,training_data[index][0],training_data[index][1], out[0]);
    }
    printf("loss: %f\n", loss/batch_size);


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

