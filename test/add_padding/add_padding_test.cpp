//
// Created by thainq on 12/15/24.
//

#include "add_padding_test.h"

void test_add_padding() {
    // Test 1: Padding bằng 0 (tensor không thay đổi)
    {
        torch::Tensor matrix = torch::tensor({{1, 2}, {3, 4}}, torch::kFloat32);
        auto padded_matrix = convolution::add_padding(matrix, {0, 0});
        std::cout << "Test 1 - Padding (0, 0):\n" << padded_matrix << std::endl;
        // Kiểm tra xem padded_matrix có bằng với matrix ban đầu không
        assert(torch::allclose(matrix, padded_matrix));
    }

    // Test 2: Padding dọc và ngang bằng 1
    {
        torch::Tensor matrix = torch::tensor({{1, 2}, {3, 4}}, torch::kFloat32);
        auto padded_matrix = convolution::add_padding(matrix, {1, 1});
        std::cout << "Test 2 - Padding (1, 1):\n" << padded_matrix << std::endl;
        // Kiểm tra kích thước của tensor sau khi padding
        assert(padded_matrix.sizes() == torch::IntArrayRef({4, 4}));
        // Kiểm tra các giá trị padding xung quanh tensor
        assert(padded_matrix[0][0].item<int>() == 0);
        assert(padded_matrix[3][3].item<int>() == 0);
        assert(padded_matrix[1][1].item<int>() == 1); // Kiểm tra giá trị ban đầu tại vị trí (1, 1)
    }

    // Test 3: Padding dọc là 2 và ngang là 1
    {
        torch::Tensor matrix = torch::tensor({{1, 2}, {3, 4}}, torch::kFloat32);
        auto padded_matrix = convolution::add_padding(matrix, {2, 1});
        std::cout << "Test 3 - Padding (2, 1):\n" << padded_matrix << std::endl;
        // Kiểm tra kích thước tensor sau khi padding
        assert(padded_matrix.sizes() == torch::IntArrayRef({6, 4}));
        // Kiểm tra padding
        assert(padded_matrix[0][0].item<int>() == 0);
        assert(padded_matrix[5][3].item<int>() == 0);
    }

    // Test 4: Padding cho tensor lớn hơn
    {
        torch::Tensor matrix = torch::tensor({{1, 2, 3}, {4, 5, 6}}, torch::kFloat32);
        auto padded_matrix = convolution::add_padding(matrix, {1, 2});
        std::cout << "Test 4 - Padding (1, 2) for a larger tensor:\n" << padded_matrix << std::endl;
        // Kiểm tra kích thước sau khi padding
        assert(padded_matrix.sizes() == torch::IntArrayRef({4, 7}));
        // Kiểm tra các giá trị padding
        assert(padded_matrix[0][0].item<int>() == 0);
        assert(padded_matrix[3][6].item<int>() == 0);
    }
}