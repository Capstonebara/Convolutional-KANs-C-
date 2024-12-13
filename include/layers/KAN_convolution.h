#ifndef KAN_CONV_H
#define KAN_CONV_H

#include <torch/torch.h>
#include "../../include/layers/KAN_linear.h"

class KAN_Convolution_Layer : public torch::nn::Module {
    public:
        KAN_Convolution_Layer(int in_channels = 1,
                              int out_channels = 1,
                              std::pair<int, int> kernel_size = {2,2},
                              std::pair<int, int> stride = {1,1},
                              std::pair<int, int> padding = {0,0},
                              std::pair<int, int> dilation = {1,1},
                              int grid_size = 5,
                              int spline_order = 3,
                              double scale_noise = 0.1,
                              double scale_base = 1.0,
                              double scale_spline = 1.0,
                              torch::nn::SiLU base_activation = torch::nn::SiLU(),
                              double grid_eps = 0.02,
                              std::pair<double, double> grid_range = {-1, 1},
                              std::string device = "cpu");
        torch::Tensor forward(const torch::Tensor& x);

    private:
        int in_channels;
        int out_channels;
        std::pair<int, int> kernel_size;
        std::pair<int, int> stride;
        std::pair<int, int> padding;
        std::pair<int, int> dilation;
        int grid_size;
        int spline_order;
        double scale_noise;
        double scale_base;
        double scale_spline;
        torch::nn::SiLU base_activation;
        double grid_eps;
        std::pair<double, double> grid_range;
        std::string device;
        torch::nn::ModuleList convs;
};

class KAN_Convolution : public torch::nn::Module {
    public:
        KAN_Convolution(
            std::pair<int, int> kernel_size = {2,2},
            std::pair<int, int> stride = {1,1},
            std::pair<int, int> padding = {0,0},
            std::pair<int, int> dilation = {1,1},
            int grid_size = 5,
            int spline_order = 3,
            double scale_noise = 0.1,
            double scale_base = 1.0,
            double scale_spline = 1.0,
            torch::nn::SiLU base_activation = torch::nn::SiLU(),
            double grid_eps = 0.02,
            std::pair<double, double> grid_range = {-1, 1},
            std::string device = "cpu");

        torch::Tensor forward(const torch::Tensor& x);
        double regularization_loss(double regularize_activation = 1.0, double regularize_entropy = 1.0);
    
    private:
        std::pair<int, int> kernel_size;
        std::pair<int, int> stride;
        std::pair<int, int> padding;
        std::pair<int, int> dilation;
        int grid_size;
        int spline_order;
        double scale_noise;
        double scale_base;
        double scale_spline;
        torch::nn::SiLU base_activation;
        double grid_eps;
        std::pair<double, double> grid_range;
        std::string device;
        KANLinear conv;
};


#endif // KAN_CONV_H