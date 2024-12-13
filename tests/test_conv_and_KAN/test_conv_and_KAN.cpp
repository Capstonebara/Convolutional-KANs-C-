#include <iostream>
#include <torch/torch.h>
#include "architectures/conv_and_KAN.h"

void print_tensor(const torch::Tensor& tensor) {
    std::cout << tensor << std::endl;
}

void test_case_1() {
    conv_and_KAN model(5);
    // model.eval();

    // Deterministic input
    auto input_tensor = torch::ones({1, 1, 28, 28});
    auto output = model.forward(input_tensor);
    std::cout << "Test Case 1 Output: ";
    print_tensor(output);
}

int main() {
    // Run test case
    test_case_1();

    return 0;
}
