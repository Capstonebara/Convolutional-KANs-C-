#include <torch/torch.h>
#include "../../include/layers/KAN_linear.h"
#include "../../include/layers/convolution.h"
#include "../../include/layers/KAN_convolution.h"

KAN_Convolution::KAN_Convolution(
    std::pair<int, int> kernel_size,
    std::pair<int, int> stride,
    std::pair<int, int> padding,
    std::pair<int, int> dilation,
    int grid_size,
    int spline_order,
    double scale_noise,
    double scale_base,
    double scale_spline,
    torch::nn::SiLU base_activation,
    double grid_eps,
    std::pair<double, double> grid_range,
    std::string device
) : kernel_size(kernel_size),
    stride(stride),
    padding(padding),
    dilation(dilation),
    grid_size(grid_size),
    spline_order(spline_order),
    device(device), 
    conv(kernel_size.first * kernel_size.second,  // in_features
         1,  // out_features
         grid_size, 
         spline_order, 
         scale_noise, 
         scale_base, 
         scale_spline, 
         true,
         base_activation, 
         grid_eps, 
         grid_range) {
};

torch::Tensor KAN_Convolution::forward(const torch::Tensor& x) {
    this->device = x.device().str();
//    torch::Tensor output = convolution::kan_conv2d(
//        x,
//        this->conv,
//        this->kernel_size,
//        this->stride,
//        this->dilation,
//        this->padding,
//        this->device
//    );
    torch::Tensor output = conv.forward(x);
    return output;
}

double KAN_Convolution::regularization_loss(double regularize_activation, double regularize_entropy) {
    //Deo hieu kieu gi
    return 0;
}


KAN_Convolution_Layer::KAN_Convolution_Layer(
    int in_channels,
    int out_channels,
    std::pair<int, int> kernel_size,
    std::pair<int, int> stride,
    std::pair<int, int> padding,
    std::pair<int, int> dilation,
    int grid_size,
    int spline_order,
    double scale_noise,
    double scale_base,
    double scale_spline,
    torch::nn::SiLU base_activation,
    double grid_eps,
    std::pair<double, double> grid_range,
    std::string device
) : in_channels(in_channels),
    out_channels(out_channels),
    kernel_size(kernel_size),
    stride(stride),
    padding(padding),
    dilation(dilation),
    grid_size(grid_size),
    spline_order(spline_order),
    device(device),
    convs(torch::nn::ModuleList()) { // Properly initialize the ModuleList

        for (int i = 0; i < in_channels * out_channels; ++i) {
            convs->push_back(std::make_shared<KAN_Convolution>( // Use std::make_shared
                kernel_size,
                stride,
                padding,
                dilation,
                grid_size,
                spline_order,
                scale_noise,
                scale_base,
                scale_spline,
                base_activation,
                grid_eps,
                grid_range,
                device
            ));
        }

    // Register the ModuleList
    register_module("convs", convs);
}

torch::Tensor KAN_Convolution_Layer::forward(const torch::Tensor& x) {
    this->device = x.device().str();

    // Convert ModuleList to std::vector<std::shared_ptr<KAN_Convolution>>
    std::vector<std::shared_ptr<KAN_Convolution>> kernel_vec;
    for (const auto& module : *convs) {
        kernel_vec.push_back(std::dynamic_pointer_cast<KAN_Convolution>(module));
    }

    torch::Tensor output = convolution::multiple_convs_kan_conv2d(
        x,                                   // Input tensor
        kernel_vec,                            // Convolution kernels
        kernel_size.first,                   // Kernel size
        out_channels,                        // Number of output channels
        stride,                              // Stride
        dilation,                            // Dilation
        padding,                             // Padding
        torch::Device(device)                // Device
    );
    return output;
}