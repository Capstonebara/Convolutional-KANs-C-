#include "layers/convolution.h"
#include <torch/torch.h>
#include <vector>
#include <tuple>
#include <iostream>

// Function definitions

std::tuple<int, int, int, int> calc_out_dims(
    const torch::Tensor& matrix,
    int kernel_side,
    const std::tuple<int, int>& stride,
    const std::tuple<int, int>& dilation,
    const std::tuple<int, int>& padding
) {
    // Extract matrix dimensions
    const auto batch_size = matrix.size(0);
    const auto n_channels = matrix.size(1);
    const auto n = matrix.size(2);
    const auto m = matrix.size(3);

    // Extract stride, dilation, and padding values
    const int stride_h = std::get<0>(stride);
    const int stride_w = std::get<1>(stride);

    const int dilation_h = std::get<0>(dilation);
    const int dilation_w = std::get<1>(dilation);

    const int padding_h = std::get<0>(padding);
    const int padding_w = std::get<1>(padding);

    // Calculate output dimensions
    const int h_out = std::floor((n + 2 * padding_h - kernel_side - (kernel_side - 1) * (dilation_h - 1)) / stride_h) + 1;
    const int w_out = std::floor((m + 2 * padding_w - kernel_side - (kernel_side - 1) * (dilation_w - 1)) / stride_w) + 1;

    return {h_out, w_out, batch_size, n_channels};
}

torch::Tensor multiple_convs_kan_conv2d() {
    torch::Tensor x;

    return x;
}


torch::Tensor add_padding(const torch::Tensor& matrix, const std::pair<int, int>& padding) {

    int r = padding.first;
    int c = padding.second;

    int n = matrix.size(0);
    int m = matrix.size(1);

    torch::Tensor padded_matrix = torch::zeros({n + 2 * r, m + 2 * c}, matrix.options());

    padded_matrix.slice(0, r, n + r).slice(1, c, m + c) = matrix;

    return padded_matrix;
}

void kan_conv2d(const torch::Tensor& x,
                const torch::Tensor& conv,
                const torch::Tensor& param,
                const std::vector<int64_t>& stride,
                const std::vector<int64_t>& dilation,
                const std::vector<int64_t>& padding,
                torch::Device device) {
    return; // return None
}