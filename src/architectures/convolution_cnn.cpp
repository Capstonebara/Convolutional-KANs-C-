#include "architectures/convolution_cnn.h"

Convolution_CNN::Convolution_CNN()
    : conv1(torch::nn::Conv2dOptions(1, 5, 3).padding({0, 0})),
      conv2(torch::nn::Conv2dOptions(5, 10, 3).padding({0, 0})),
      maxpool(torch::nn::MaxPool2dOptions(2)),
      flatten(torch::nn::FlattenOptions()),
      linear1(torch::nn::LinearOptions(250, 10)) {
    // Register modules
    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("maxpool", maxpool);
    register_module("flatten", flatten);
    register_module("linear1", linear1);
}

torch::Tensor Convolution_CNN::forward(torch::Tensor x) {
    x = torch::relu(conv1(x));
    x = maxpool(x);
    x = torch::relu(conv2(x));
    x = maxpool(x);
    x = flatten(x);
    x = linear1(x);
    x = torch::log_softmax(x, 1);
    return x;
}




