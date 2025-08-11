#include "NN.h"
#include "RNG.h"

#include <malloc.h>
#include <stdio.h>

#ifdef _WIN32
#   include <windows.h>
#   define SLEEP(msecs) Sleep(msecs)
#elif __unix
#   include <time.h>
#   define SLEEP(msecs) do { struct timespec ts; ts.tv_sec = msecs/1000; ts.tv_nsec = (msecs%1000)*1000000; nanosleep(&ts, NULL); } while (0)
#else
#   error "Unknown system"
#endif



//#define NEURONS_PER_LAYER 512
#define NEURONS_PER_LAYER 2
#define LAYERS 2

#define RANDOM_INIT_MAX 0.1
#define RANDOM_INIT_MIN -0.1

#define LEARNING_RATE 0.1
#define LEARNING_TEST_SPLIT 0.7


void shuffle(unsigned int *array, size_t n) {
    for (size_t i = n - 1; i > 0; i--) {
        size_t j = rand() % (i + 1);
        unsigned int tmp = array[i];
        array[i] = array[j];
        array[j] = tmp;
    }
}


void main() {

    // settings
    NN_learning_settings* learning_settings = (NN_learning_settings*)malloc(sizeof(NN_learning_settings));
    NN_use_settings* use_settings = (NN_use_settings*)malloc(sizeof(NN_use_settings));
    learning_settings->learning_rate = LEARNING_RATE;
    use_settings->activation = SIGMOID;
    learning_settings->optimizer = ADAMW;
    learning_settings->use_batching = true;
    use_settings->device_type = CPU;

    // init
    unsigned int neurons_per_layer[2] = {2,2};
    NN_network* net = NN_network_init(neurons_per_layer, 2);
    NN_trainer* trainer = NN_trainer_init(net, learning_settings, use_settings, "cpu1");
    
    if (NN_network_load_from_file(net, "sample_net.net")==-3) {
        // if no network to load from exists, just randomise
        NN_network_randomise_xaivier(net, RANDOM_INIT_MIN, RANDOM_INIT_MAX);
    }
    

    // train
    unsigned int batch_size = 8;
    float training_data[8][2][2] = {
        {{0,0},{0,0}},
        {{0,1},{0,1}},
        {{1,0},{1,0}},
        {{1,1},{1,1}},
        {{0.1,0.1},{0.1,0.1}},
        {{0,0.5},{0,0.5}},
        {{0.5,0},{0.5,0}},
        {{0.9,0.9},{0.9,0.9}},
    };

    unsigned int idx[8] = {0,1,2,3,4,5,6,7};

    unsigned int epoch = 0;
    float loss = 0;
    for (unsigned int i = 0; i < 40; i++) {
        shuffle(idx, 8);
        loss = 0;
        for (unsigned int batch = 0; batch < batch_size; batch++) {
            unsigned int index = idx[batch];
            NN_trainer_accumulate(trainer, training_data[index][0],training_data[index][1]);
            loss += NN_trainer_loss(trainer, training_data[index][1]);
        }
        printf("epoch %i, loss: %f\n", epoch, loss/batch_size);
        NN_trainer_apply(trainer, batch_size);
        epoch++;
    }
    
    // save to file
    printf("saving network...\n");
    NN_network_save_to_file(net, "sample_net.net");
    
    

    // clean up
    printf("cleaning up...\n");
    NN_trainer_free(trainer);
    NN_network_free(net);
    free(learning_settings);
    free(use_settings);

}
