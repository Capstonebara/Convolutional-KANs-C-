#include "layers/convolution.h"
#include <torch/torch.h>
#include <layers/KAN_convolution.h>
#include <vector>
#include <tuple>
#include <iostream>

// Function definitions

std::tuple<int, int, int, int> convolution::calc_out_dims(
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

torch::Tensor convolution::multiple_convs_kan_conv2d(
    const torch::Tensor& matrix,
    const std::vector<std::shared_ptr<KAN_Convolution>>& kernels,
    int kernel_side,
    int out_channels,
    const std::pair<int, int>& stride,
    const std::pair<int, int>& dilation,
    const std::pair<int, int>& padding,
    const torch::Device& device
) {

    auto [h_out, w_out, batch_size, in_channels] = calc_out_dims(matrix, kernel_side, stride, dilation, padding);

    auto matrix_out = torch::zeros({batch_size, out_channels, h_out, w_out}, torch::TensorOptions().device(device));

    auto unfold = torch::nn::Unfold(torch::nn::UnfoldOptions({kernel_side, kernel_side})
                                       .dilation({dilation.first, dilation.second})
                                       .padding({padding.first, padding.second})
                                       .stride({stride.first, stride.second}));

    auto conv_groups = unfold(matrix)
                          .view({batch_size, in_channels, kernel_side * kernel_side, h_out * w_out})
                          .transpose(2, 3);

    int kern_per_out = kernels.size() / out_channels;

    for (int c_out = 0; c_out < out_channels; ++c_out) {
        auto out_channel_accum = torch::zeros({batch_size, h_out, w_out}, torch::TensorOptions().device(device));

        for (int k_idx = 0; k_idx < kern_per_out; ++k_idx) {
            const auto& kernel = kernels[c_out * kern_per_out + k_idx];
            auto conv_input = conv_groups.narrow(2, k_idx, 1).flatten(0, 1);
            auto conv_result = kernel->forward(conv_groups.index({torch::indexing::Slice(), k_idx, torch::indexing::Slice(), torch::indexing::Slice()})
                                                  .flatten(0, 1));
            out_channel_accum += conv_result.view({batch_size, h_out, w_out});
        }
        matrix_out.index_put_({torch::indexing::Slice(), c_out, torch::indexing::Slice(), torch::indexing::Slice()}, out_channel_accum);
    }
    return matrix_out;
}


torch::Tensor convolution::add_padding(const torch::Tensor& matrix, const std::pair<int, int>& padding) {

    int r = padding.first;
    int c = padding.second;

    int n = matrix.size(0);
    int m = matrix.size(1);

    torch::Tensor padded_matrix = torch::zeros({n + 2 * r, m + 2 * c}, matrix.options());

    padded_matrix.slice(0, r, n + r).slice(1, c, m + c) = matrix;

    return padded_matrix;
}

// void kan_conv2d(const torch::Tensor& x,
//                 const torch::Tensor& conv,
//                 const torch::Tensor& param,
//                 const std::vector<int64_t>& stride,
//                 const std::vector<int64_t>& dilation,
//                 const std::vector<int64_t>& padding,
//                 torch::Device device) {
//     return; // return None
// }

// Thai oi, Dung gen ra cho nay de test ham thoi nhe // ok bac
torch::Tensor convolution::kan_conv2d(
        const torch::Tensor& x,
        const KANLinear& conv,  // This matches the modified header
        const std::pair<int, int>& kernel_size,
        const std::pair<int, int>& stride,
        const std::pair<int, int>& dilation,
        const std::pair<int, int>& padding,
        const std::string& device) {
        // Create a dummy tensor with the same shape as input `x` for testing purposes.
        torch::Tensor dummy_output = torch::zeros_like(x, torch::TensorOptions().device(device));
        return dummy_output; // Return the dummy tensor
    }