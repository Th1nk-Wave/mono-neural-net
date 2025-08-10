*** super high priority
[] add endian handling to file saving/loading logic



*** first things first
[] init functions for network, trainer and processor.
[] free functions for above.
[] implement basic RELU and Gradient Descent, sucsefully train a basic 2 layer model.

*** next steps
[] write optimization functions for ADAM and Stochastic Gradient Descent.
[] train network again using updated optimization functions.
[] basic multithreading to train multiple networks at the same time.
[] compare previous loss/time with new optimized loss/time.

*** nice-to-have
[] areana allocator for network
[] write a basic compute shader that can use a basic network.
[] support loading and writing neural network data with a file.
[] turn into a portable lib for future use.

*** going the extra mile
[] support multiple types of neural networks: convolutional, spiking network, transformer etc..
[] enable amd implement GPU hardware acceleration.