#ifndef KANCONV_AND_KAN_H
#define KANCONV_AND_KAN_H


#include <torch/torch.h>
#include "../../include/layers/KAN_convolution.h"
#include "../../include/layers/KAN_linear.h"

class KANconv_and_KAN : public torch::nn::Module {
public:
    KANconv_and_KAN(int grid_size = 5);
    torch::Tensor forward(torch::Tensor x);

private:
    KAN_Convolution_Layer conv1;
    KAN_Convolution_Layer conv2;
    torch::nn::MaxPool2d pool1{nullptr};
    torch::nn::Flatten flatten{nullptr};
    KANLinear kan1;
};

#endif //KANCONV_AND_KAN_H
