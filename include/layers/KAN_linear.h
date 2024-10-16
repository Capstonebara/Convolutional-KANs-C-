#ifndef KAN_H
#define KAN_H

#include <torch/torch.h>
#include <vector>

// KANLinear class definition
struct KANLinear : torch::nn::Module {
    int64_t in_features, out_features, grid_size, spline_order;
    float scale_noise, scale_base, scale_spline, grid_eps;
    bool enable_standalone_scale_spline;
    torch::Tensor grid, base_weight, spline_weight, spline_scaler;
    torch::nn::SiLU base_activation;

    // Constructor
    KANLinear(int64_t in_features, int64_t out_features, int64_t grid_size = 5, int64_t spline_order = 3, 
              float scale_noise = 0.1, float scale_base = 1.0, float scale_spline = 1.0, bool enable_standalone_scale_spline = true,
              float grid_eps = 0.02, std::vector<float> grid_range = {-1.0, 1.0});

    // Methods
    void reset_parameters();
    torch::Tensor b_splines(torch::Tensor x);
    torch::Tensor curve2coeff(torch::Tensor x, torch::Tensor y);
    torch::Tensor scaled_spline_weight();
    torch::Tensor forward(torch::Tensor x);
};

// KAN class definition
struct KAN : torch::nn::Module {
    std::vector<KANLinear> layers;

    // Constructor
    KAN(std::vector<int64_t> layers_hidden, int64_t grid_size = 5, int64_t spline_order = 3, float scale_noise = 0.1, float scale_base = 1.0, 
        float scale_spline = 1.0, float grid_eps = 0.02, std::vector<float> grid_range = {-1.0, 1.0});

    // Methods
    torch::Tensor forward(torch::Tensor x, bool update_grid = false);
    torch::Tensor regularization_loss(float regularize_activation = 1.0, float regularize_entropy = 1.0);
};

#endif // KAN_H
