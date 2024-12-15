#include "architectures/KANconv_and_MLP.h"

void KANconv_and_MLP_test() {
    int seed = 42;
    torch::manual_seed(seed);

    KANC_MLP model(5);
    // model.eval();

    // Deterministic input
    auto input_tensor = torch::ones({1, 1, 28, 28});
    auto output = model.forward(input_tensor);
    std::cout << "Test Case 1 Output: ";
    std::cout << output << std::endl;
    std::cout << "Expected Output: " << "[-2.5086 -2.0284 -2.2649 -2.1895 -2.4841 -2.2020 -2.3526 -2.4853 -2.5463 -2.1162]" << std::endl;
}
