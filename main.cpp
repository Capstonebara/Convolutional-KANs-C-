// #include <torch/torch.h>
// #include <iostream>
// #include <layers/convolution.h>
// #include <layers/KAN_linear.h>
// #include <layers/KAN_convolution.h>

// int main() {
//   torch::Tensor tensor = torch::rand({2, 3});
//   std::cout << tensor << std::endl;

// }

// Thai oi, cai nay lai de test thoi nhe
#include <torch/torch.h>
#include <iostream>
#include <memory>  // For std::make_shared
#include <layers/convolution.h>
#include <layers/KAN_linear.h>
#include <layers/KAN_convolution.h>

int main() {
    // Define parameters for the KAN_Convolution_Layer
    int in_channels = 3;
    int out_channels = 2;
    std::pair<int, int> kernel_size = {3, 3};
    std::pair<int, int> stride = {1, 1};
    std::pair<int, int> padding = {1, 1};
    std::pair<int, int> dilation = {1, 1};
    int grid_size = 4;
    int spline_order = 3;
    double scale_noise = 0.1;
    double scale_base = 0.2;
    double scale_spline = 0.3;
    torch::nn::SiLU base_activation = torch::nn::SiLU();
    double grid_eps = 1e-5;
    std::pair<double, double> grid_range = {0.0, 1.0};
    std::string device = "cpu";  // or "cuda" if you want to use a GPU

    // Create an instance of KAN_Convolution_Layer
    KAN_Convolution_Layer conv_layer(
        in_channels, out_channels, kernel_size, stride, padding, dilation,
        grid_size, spline_order, scale_noise, scale_base, scale_spline,
        base_activation, grid_eps, grid_range, device
    );

    // Create a random tensor with appropriate dimensions (batch_size, channels, height, width)
    // Let's assume batch size of 2, 3 input channels, and a 4x4 image
    torch::Tensor matrix = torch::tensor({
        { { {0.1, 0.2, 0.3, 0.4, 0.5, 0.6},
            {0.7, 0.8, 0.9, 1.0, 1.1, 1.2},
            {1.3, 1.4, 1.5, 1.6, 1.7, 1.8},
            {1.9, 2.0, 2.1, 2.2, 2.3, 2.4},
            {2.5, 2.6, 2.7, 2.8, 2.9, 3.0},
            {3.1, 3.2, 3.3, 3.4, 3.5, 3.6} },

          { {0.5, 0.4, 0.3, 0.2, 0.1, 0.0},
            {0.9, 0.8, 0.7, 0.6, 0.5, 0.4},
            {1.3, 1.2, 1.1, 1.0, 0.9, 0.8},
            {1.7, 1.6, 1.5, 1.4, 1.3, 1.2},
            {2.1, 2.0, 1.9, 1.8, 1.7, 1.6},
            {2.5, 2.4, 2.3, 2.2, 2.1, 2.0} },

          { {0.6, 0.7, 0.8, 0.9, 1.0, 1.1},
            {1.2, 1.3, 1.4, 1.5, 1.6, 1.7},
            {1.8, 1.9, 2.0, 2.1, 2.2, 2.3},
            {2.4, 2.5, 2.6, 2.7, 2.8, 2.9},
            {3.0, 3.1, 3.2, 3.3, 3.4, 3.5},
            {3.6, 3.7, 3.8, 3.9, 4.0, 4.1} }
        }
    });  // [batch_size, channels, height, width]

    // Print the input tensor
    // std::cout << "Input Tensor: " << tensor << std::endl;

    // Pass the tensor through the KAN_Convolution_Layer
    torch::Tensor output = conv_layer.forward(matrix);

    // Print the output tensor
    std::cout << "Output Tensor Shape: " << output.sizes() << std::endl;

    return 0;
}
