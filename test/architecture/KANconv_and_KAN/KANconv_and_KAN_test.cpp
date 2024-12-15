
#include "KANconv_and_KAN_test.h"

void KANconv_and_KAN_test() {
    int seed = 42;
    torch::manual_seed(seed);

    KANconv_and_KAN model(5);
    // model.eval();

    // Deterministic input
    auto input_tensor = torch::ones({1, 1, 28, 28});
    auto output = model.forward(input_tensor);
    std::cout << "Test Case 1 Output: ";
    std::cout << output << std::endl;
    std::cout << "Expected Output: " << "[-2.4076 -2.1574 -2.2863 -2.2353 -2.3554 -2.2435 -2.3234 -2.3756 -2.4425 -2.2353]" << std::endl;
}