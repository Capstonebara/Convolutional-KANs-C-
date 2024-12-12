#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <torch/torch.h>
#include <vector>
#include <tuple>
#include "../../include/layers/KAN_linear.h"

namespace convolution {
// Function declarations
std::tuple<int, int, int, int> calc_out_dims(
    const torch::Tensor& matrix,
    int kernel_side,
    const std::tuple<int, int>& stride,
    const std::tuple<int, int>& dilation,
    const std::tuple<int, int>& padding
);

torch::Tensor add_padding(const torch::Tensor& matrix, const std::pair<int, int>& padding);

torch::Tensor multiple_convs_kan_conv2d();

torch::Tensor kan_conv2d(
        const torch::Tensor& x,
        const KANLinear& conv,  // Change this to KANLinear, which matches the implementation
        const std::pair<int, int>& kernel_size = {2, 2},
        const std::pair<int, int>& stride = {1, 1},
        const std::pair<int, int>& dilation = {1, 1},
        const std::pair<int, int>& padding = {0, 0},
        const std::string& device = "cpu"
    );

};



#endif // CONVOLUTION_H
