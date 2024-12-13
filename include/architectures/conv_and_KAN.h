#ifndef CONV_AND_KAN_H
#define CONV_AND_KAN_H

#include <torch/torch.h>
#include "../../include/layers/KAN_linear.h"

class conv_and_KAN : public torch::nn::Module {
public:
    conv_and_KAN(int grid_size = 5);

    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::Conv2d conv2{nullptr};
    torch::nn::MaxPool2d maxpool{nullptr};
    torch::nn::Flatten flatten{nullptr};
    KANLinear kan1;
};

#endif // CONV_AND_KAN_H
