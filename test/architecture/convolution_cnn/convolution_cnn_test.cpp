
#include "convolution_cnn_test.h"

void convolution_cnn_test() {
    int seed = 42;
    torch::manual_seed(seed);

    Convolution_CNN model{};
    // model.eval();

    // Deterministic input
    auto input_tensor = torch::ones({1, 1, 28, 28});
    auto output = model.forward(input_tensor);
    std::cout << "Test Case 1 Output: ";
    std::cout << output << std::endl;
    std::cout << "Expected Output: " << "[-2.3761 -2.4246 -2.3203 -2.2732 -2.0478 -2.5287 -2.4780 -2.3334 -1.9903 -2.3967]" << std::endl;
}