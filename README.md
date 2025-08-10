# end this maddness please


# network file specification
[2 byte] version
[1 byte] activation func id (currently unused)
[4 byte] layers
[layers * 4 bytes] neurons_per_layer (array)
[\sum (float32 * neurons_per_layer[i] * neurons_per_layer[i+1]) bytes] weights (double nested array) (float*** weights[layer][neuron][connection])
[\sum (float32 * neurons_per_layer[i]) bytes] bias (nested array) (float** bias[layer][neuron])