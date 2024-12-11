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

at::Tensor multiple_convs_kan_conv2d(
    const at::Tensor& matrix,
    const std::vector<std::shared_ptr<torch::nn::Conv2d>>& kernels,
    int64_t kernel_side,
    int64_t out_channels,
    const std::pair<int64_t, int64_t>& stride = {1, 1},
    const std::pair<int64_t, int64_t>& dilation = {1, 1},
    const std::pair<int64_t, int64_t>& padding = {0, 0},
    const torch::Device& device = torch::kCUDA);

torch::Tensor add_padding(const torch::Tensor& matrix, const std::pair<int, int>& padding);

#endif // CONVOLUTION_H
