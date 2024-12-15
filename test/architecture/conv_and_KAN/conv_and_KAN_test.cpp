#include "conv_and_KAN_test.h"

void conv_and_KAN_test() {
    int seed = 42;
    torch::manual_seed(seed);

    conv_and_KAN model(5);
    // model.eval();

    // Deterministic input
    auto input_tensor = torch::ones({1, 1, 28, 28});
    auto output = model.forward(input_tensor);
    std::cout << "Test Case 1 Output: ";
    std::cout << output << std::endl;
    std::cout << "Expected Output: " << "[-2.2961 -2.3912 -2.3067 -2.3099 -2.1374 -2.3952 -2.4114 -2.3441 -2.1228 -2.3587]" << std::endl;
}