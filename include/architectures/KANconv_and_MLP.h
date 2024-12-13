#ifndef KAN_CONV_MLP
#define KAN_CONV_MLP

#include <torch/torch.h>
#include "../../include/layers/KAN_convolution.h"

class KANC_MLP : public torch::nn::Module {
    public:
        KANC_MLP(int grid_size = 5);
        torch::Tensor forward(torch::Tensor x);

    private:
        KAN_Convolution_Layer conv1;
        KAN_Convolution_Layer conv2;
        torch::nn::MaxPool2d pool1{nullptr};
        torch::nn::Flatten flatten{nullptr};
        torch::nn::Linear linear1{nullptr};
};

#endif // KAN_CONV_MLP