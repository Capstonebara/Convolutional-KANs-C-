#include "architectures/conv_and_KAN.h"

conv_and_KAN::conv_and_KAN(int grid_size)
    : conv1(torch::nn::Conv2dOptions(1, 5, 3).padding({0, 0})),
      conv2(torch::nn::Conv2dOptions(5, 10, 3).padding({0, 0})),
      maxpool(torch::nn::MaxPool2dOptions(2)),
      flatten(torch::nn::FlattenOptions()),
      kan1(250, 10, grid_size, 3, 0.01, 1, 1, true, torch::nn::SiLU(), 0.02, {0, 1}) {
    // Register modules
    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("maxpool", maxpool);
    register_module("flatten", flatten);
    register_module("kan1", torch::nn::ModuleHolder<KANLinear>(kan1));
}

torch::Tensor conv_and_KAN::forward(torch::Tensor x) {
    x = torch::relu(conv1(x));
    x = maxpool(x);
    x = torch::relu(conv2(x));
    x = maxpool(x);
    x = flatten(x);
    x = kan1.forward(x);
    x = torch::log_softmax(x, 1);
    return x;
}
