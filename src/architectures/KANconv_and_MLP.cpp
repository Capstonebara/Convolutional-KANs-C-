#include "architectures/KANconv_and_MLP.h"

KANC_MLP::KANC_MLP(int grid_size)
    : conv1(1, 5, {3, 3}, {1, 1}, {0, 0}, {1, 1}, grid_size, 3, 0.1, 1.0, 1.0, torch::nn::SiLU(), 0.02, {-1, 1}, "cpu"),
      conv2(5, 5, {3, 3}, {1, 1}, {0, 0}, {1, 1}, grid_size, 3, 0.1, 1.0, 1.0, torch::nn::SiLU(), 0.02, {-1, 1}, "cpu"),
      pool1(torch::nn::MaxPool2dOptions(2)),
      flatten(torch::nn::FlattenOptions()),
      linear1(torch::nn::LinearOptions(250, 10)) {
    // Register modules
    register_module("conv1", torch::nn::ModuleHolder<KAN_Convolution_Layer>(conv1));
    register_module("conv2", torch::nn::ModuleHolder<KAN_Convolution_Layer>(conv2));
    register_module("pool1", pool1);
    register_module("flatten", flatten);
    register_module("linear1", linear1);
}

torch::Tensor KANC_MLP::forward(torch::Tensor x) {
    x = conv1.forward(x);
    x = pool1(x);
    x = conv2.forward(x);
    x = pool1(x);
    x = flatten(x);
    x = linear1(x);
    x = torch::log_softmax(x, 1);
    return x;
}