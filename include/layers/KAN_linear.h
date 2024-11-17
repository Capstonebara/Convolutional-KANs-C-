#ifndef KAN_LINEAR_H
#define KAN_LINEAR_H

#include </home/nhomnhom0/miniforge3/pkgs/pytorch-2.5.1-py3.9_cpu_0/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/torch/torch.h>

class KANLinear : public torch::nn::Module {
public:
    // Constructor with default values for parameters
    KANLinear(int in_features, int out_features, int grid_size = 5, int spline_order = 3,
              double scale_noise = 0.1, double scale_base = 1.0, double scale_spline = 1.0,
              bool enable_standalone_scale_spline = true, torch::nn::SiLU base_activation = torch::nn::SiLU(),
              double grid_eps = 0.02, std::pair<double, double> grid_range = {-1, 1});

    void reset_parameters(); // Initializes parameters
    torch::Tensor b_splines(torch::Tensor& x); // B-spline computation
    torch::Tensor curve2coeff(torch::Tensor& x, torch::Tensor& y); // Interpolation coefficients
    torch::Tensor forward(const torch::Tensor& x); // Forward pass
    void update_grid(const torch::Tensor& x, double margin = 0.01); // Updates the grid
    torch::Tensor regularization_loss(double regularize_activation = 1.0, double regularize_entropy = 1.0); // Loss computation

    // Getter functions to access private members
    torch::Tensor get_base_weight() const { return base_weight; }
    torch::Tensor get_spline_weight() const { return spline_weight; }
    torch::Tensor get_spline_scaler() const { return spline_scaler; }
    bool is_enable_standalone_scale_spline() const { return enable_standalone_scale_spline; }

private:
    int in_features; // Number of input features
    int out_features; // Number of output features
    int grid_size; // Size of the grid for B-splines
    int spline_order; // Order of the spline
    double scale_noise; // Noise scaling factor
    double scale_base; // Base scaling factor
    double scale_spline; // Spline scaling factor
    bool enable_standalone_scale_spline; // Flag to enable standalone scaling
    double grid_eps; // Grid epsilon value
    torch::nn::SiLU base_activation; // Base activation function

    // Private member tensors
    torch::Tensor grid; // Grid tensor for B-splines
    torch::Tensor base_weight; // Base weight tensor
    torch::Tensor spline_weight; // Spline weight tensor
    torch::Tensor spline_scaler; // Spline scaler tensor (if enabled)
};

#endif // KAN_LINEAR_H
