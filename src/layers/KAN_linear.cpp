#include "../../include/layers/KAN_linear.h"
#include <torch/torch.h>
#include <iostream>

KANLinear::KANLinear(int in_features, int out_features, int grid_size, int spline_order,
                     double scale_noise, double scale_base, double scale_spline,
                     bool enable_standalone_scale_spline, torch::nn::SiLU base_activation,
                     double grid_eps, std::pair<double, double> grid_range)
    : in_features(in_features), out_features(out_features), grid_size(grid_size), spline_order(spline_order),
      scale_noise(scale_noise), scale_base(scale_base), scale_spline(scale_spline),
      enable_standalone_scale_spline(enable_standalone_scale_spline), grid_eps(grid_eps),
      base_activation(base_activation) {

    // Validate input parameters for sanity
    if (grid_size <= 0 || spline_order < 0) {
        throw std::invalid_argument("grid_size must be > 0 and spline_order >= 0");
    }
    if (grid_range.first >= grid_range.second) {
        throw std::invalid_argument("grid_range.first must be less than grid_range.second");
    }

    // Initialize grid
    double h = (grid_range.second - grid_range.first) / grid_size;
    grid = torch::arange(-spline_order, grid_size + spline_order + 1) * h + grid_range.first;
    grid = grid.expand({in_features, -1}).contiguous();
    // std::cout << "Grid shape: " << grid.sizes() << std::endl;
    register_buffer("grid", grid);

    // Initialize weights
    base_weight = register_parameter("base_weight", torch::empty({out_features, in_features}));
    spline_weight = register_parameter("spline_weight", torch::empty({out_features, in_features, grid_size + spline_order}));
    if (enable_standalone_scale_spline) {
        spline_scaler = register_parameter("spline_scaler", torch::empty({out_features, in_features}));
    }

    reset_parameters(); // Call reset_parameters to initialize weights
}

void KANLinear::reset_parameters() {
    // Initialize the base weight using Kaiming uniform
    torch::nn::init::kaiming_uniform_(base_weight, std::sqrt(5.0) * scale_base);

    // No-gradient block for noise generation
    {
        torch::NoGradGuard no_grad;
        torch::Tensor noise = (torch::rand({grid_size + 1, in_features, out_features}) - 0.5) * scale_noise / grid_size;
        // std::cout << "Noise size: " << noise.sizes() << std::endl;

        auto scale_factor = enable_standalone_scale_spline ? 1.0 : scale_spline;
        auto adjusted_grid = grid.transpose(0, 1).slice(0, spline_order, -spline_order);
        auto new_spline_weight = scale_factor *  curve2coeff(adjusted_grid, noise);
        // std::cout << "new_spline_weight size: " << new_spline_weight.sizes() << std::endl;
        spline_weight.copy_(new_spline_weight);

        if (enable_standalone_scale_spline) {
            // Kaiming uniform initialization (equivalent to torch.nn.init.kaiming_uniform_)
            torch::nn::init::kaiming_uniform_(spline_scaler, std::sqrt(5.0) * scale_spline);
        };
    };
}

torch::Tensor KANLinear::b_splines (torch::Tensor& x) {
    // // Assert that the tensor has the expected dimensions (batch_size, in_features)
    assert(x.dim() == 2 && x.size(1) == in_features);

    torch::Tensor grid = this->grid; // Shape: (in_features, grid_size + 2 * spline_order + 1)
    x = x.unsqueeze(-1);
    torch::Tensor bases = (x >= grid.slice(1, 0, grid.size(1) - 1)) & (x < grid.slice(1, 1, grid.size(1)));
    bases = bases.to(x.dtype());

    for (int k = 1; k <= spline_order; ++k) {
        torch::Tensor left_part = (x - grid.slice(1, 0, grid.size(1) - (k + 1))) / (grid.slice(1, k, grid.size(1) - 1) - grid.slice(1, 0, grid.size(1) - (k + 1))) * bases.slice(2, 0, bases.size(2) - 1);
        torch::Tensor right_part = (grid.slice(1, k + 1, grid.size(1)) - x)  / (grid.slice(1, k + 1, grid.size(1)) - grid.slice(1, 1, grid.size(1) - k)) * bases.slice(2, 1, bases.size(2));
        bases = left_part + right_part;
    }

    assert(bases.sizes()[0] == x.sizes()[0]);
    assert(bases.sizes()[1] == in_features);
    assert(bases.sizes()[2] == grid_size + spline_order);

    return bases.contiguous();

    // return torch::zeros({x.size(0), out_features});

}


torch::Tensor KANLinear::curve2coeff(torch::Tensor& x, torch::Tensor& y) {
    assert(x.ndimension() == 2);
    assert(x.sizes()[1] == in_features);
    assert(y.sizes()[0] == x.sizes()[0]);
    assert(y.sizes()[1] == in_features);
    assert(y.sizes()[2] == out_features);

    // Calculate B-splines and transpose
    torch::Tensor A = b_splines(x);  // Shape: [batch_size, in_features, grid_size + spline_order]
    // std::cout << "A shape after b_splines: " << A.sizes() << std::endl;
    
    A = A.transpose(0, 1);  // Shape: [in_features, batch_size, grid_size + spline_order]
    torch::Tensor B = y.transpose(0, 1);  // Shape: [in_features, batch_size, out_features]
    // std::cout << "B shapes " << B.sizes() << std::endl;

    // auto lstsq_result = torch::linalg::lstsq(A, B, 1e-10, nullptr);
    // std::cout << "done lstsq"<< std::endl;
    // auto solution = std::get<0>(lstsq_result);  // Shape: [in_features, grid_size + spline_order, out_features]
    
    //using torch::linalg::pinv instead of torch::linalg::lstsq
    torch::Tensor pseudo_inverse = torch::linalg::pinv(A);
    torch::Tensor solution = torch::matmul(pseudo_inverse, B);

    // // Permute to get final shape
    torch::Tensor result = solution.permute({2, 0, 1});  // Shape: [out_features, in_features, grid_size + spline_order]
    
    assert(result.size(0) == out_features);
    assert(result.size(1) == in_features);
    assert(result.size(2) == grid_size + spline_order);
    
    return result.contiguous();
    // return torch::zeros({out_features, in_features, grid_size + spline_order});

}

torch::Tensor KANLinear::forward(const torch::Tensor& x) {
    return torch::zeros({x.size(0), out_features});
}

void KANLinear::update_grid(const torch::Tensor& x, double margin) {
    // No-op, for now
}

torch::Tensor KANLinear::regularization_loss(double regularize_activation, double regularize_entropy) {
    return torch::tensor(0.0); // No-op, for now
}
