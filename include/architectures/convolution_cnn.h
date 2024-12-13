#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <torch/torch.h>

class Convolution_CNN : public torch::nn::Module {
public:
    Convolution_CNN();
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::Conv2d conv2{nullptr};
    torch::nn::MaxPool2d maxpool{nullptr};
    torch::nn::Flatten flatten{nullptr};
    torch::nn::Linear linear1{nullptr};
};

#endif //CONVOLUTION_H
