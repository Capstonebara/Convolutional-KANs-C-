#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <torch/torch.h>
#include <vector>
#include <tuple>

// Function declarations
std::tuple<int, int, int, int> calc_out_dims(
    const torch::Tensor& matrix,
    int kernel_side,
    const std::tuple<int, int>& stride,
    const std::tuple<int, int>& dilation,
    const std::tuple<int, int>& padding
);

torch::Tensor multiple_convs_kan_conv2d(
    const torch::Tensor& matrix,
    const std::vector<std::shared_ptr<torch::nn::Module>>& kernels,
    int kernel_side,
    int out_channels,
    const std::tuple<int, int>& stride = {1, 1},
    const std::tuple<int, int>& dilation = {1, 1},
    const std::tuple<int, int>& padding = {0, 0},
    const std::string& device = "cuda"
);

torch::Tensor add_padding(
    const torch::Tensor& matrix,
    const std::tuple<int, int>& padding
);

#endif // CONVOLUTION_H
