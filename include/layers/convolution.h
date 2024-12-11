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

torch::Tensor add_padding(const torch::Tensor& matrix, const std::pair<int, int>& padding);

#endif // CONVOLUTION_H
