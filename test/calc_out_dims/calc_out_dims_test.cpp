//
// Created by thainq on 12/15/24.
//

#include "calc_out_dims_test.h"

void func_calc_out_dims(torch::Tensor matrix, int kernel_side, std::tuple<int, int> stride,
                  std::tuple<int, int> dilation, std::tuple<int, int> padding,
                  int expected_h_out, int expected_w_out) {
    auto [h_out, w_out, batch_size, n_channels] = convolution::calc_out_dims(matrix, kernel_side, stride, dilation, padding);
    std::cout << "Expected Output: h_out = " << expected_h_out << ", w_out = " << expected_w_out << std::endl;
    std::cout << "Actual Output: h_out = " << h_out << ", w_out = " << w_out << std::endl;
    if (h_out == expected_h_out && w_out == expected_w_out) {
        std::cout << "Test Passed!" << std::endl;
    } else {
        std::cout << "Test Failed!" << std::endl;
    }
}

void testcase_calc_out_dims() {
    // Test Case 1: Basic case
    torch::Tensor matrix1 = torch::ones({1, 1, 6, 6}); // batch_size=1, n_channels=1, height=6, width=6
    int kernel_size1 = 3;
    std::tuple<int, int> stride1 = {1, 1};
    std::tuple<int, int> dilation1 = {1, 1};
    std::tuple<int, int> padding1 = {1, 1};
    func_calc_out_dims(matrix1, kernel_size1, stride1, dilation1, padding1, 6, 6); // Expected: h_out = 6, w_out = 6

    // Test Case 2: Stride of 2
    torch::Tensor matrix2 = torch::ones({1, 1, 6, 6});
    int kernel_size2 = 3;
    std::tuple<int, int> stride2 = {2, 2};
    std::tuple<int, int> dilation2 = {1, 1};
    std::tuple<int, int> padding2 = {1, 1};
    func_calc_out_dims(matrix2, kernel_size2, stride2, dilation2, padding2, 3, 3); // Expected: h_out = 3, w_out = 3

    // Test Case 3: Larger padding
    torch::Tensor matrix3 = torch::ones({1, 1, 6, 6});
    int kernel_size3 = 3;
    std::tuple<int, int> stride3 = {1, 1};
    std::tuple<int, int> dilation3 = {1, 1};
    std::tuple<int, int> padding3 = {2, 2};
    func_calc_out_dims(matrix3, kernel_size3, stride3, dilation3, padding3, 8, 8); // Expected: h_out = 8, w_out = 8

    // Test Case 4: Dilation of 2
    torch::Tensor matrix4 = torch::ones({1, 1, 6, 6});
    int kernel_size4 = 3;
    std::tuple<int, int> stride4 = {1, 1};
    std::tuple<int, int> dilation4 = {2, 2};
    std::tuple<int, int> padding4 = {1, 1};
    func_calc_out_dims(matrix4, kernel_size4, stride4, dilation4, padding4, 4, 4); // Expected: h_out = 4, w_out = 4
}