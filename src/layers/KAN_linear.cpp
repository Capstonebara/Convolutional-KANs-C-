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
        this->spline_weight.data().copy_(new_spline_weight);

        if (enable_standalone_scale_spline) {
            // Kaiming uniform initialization (equivalent to torch.nn.init.kaiming_uniform_)
            torch::nn::init::kaiming_uniform_(spline_scaler, std::sqrt(5.0) * scale_spline);
        };
    };
}

torch::Tensor KANLinear::b_splines (const torch::Tensor& x) {
    // // Assert that the tensor has the expected dimensions (batch_size, in_features)
    assert(x.dim() == 2 && x.size(1) == in_features);

    // Create a copy of x to avoid modifying the original tensor
    torch::Tensor x_copy = x.clone();

    torch::Tensor grid = this->grid; // Shape: (in_features, grid_size + 2 * spline_order + 1)
    x_copy = x_copy.unsqueeze(-1);
    torch::Tensor bases = (x_copy >= grid.slice(1, 0, grid.size(1) - 1)) & (x_copy < grid.slice(1, 1, grid.size(1)));
    bases = bases.to(x_copy.dtype());

    for (int k = 1; k <= spline_order; ++k) {
        torch::Tensor left_part = (x_copy - grid.slice(1, 0, grid.size(1) - (k + 1))) / (grid.slice(1, k, grid.size(1) - 1) - grid.slice(1, 0, grid.size(1) - (k + 1))) * bases.slice(2, 0, bases.size(2) - 1);
        torch::Tensor right_part = (grid.slice(1, k + 1, grid.size(1)) - x_copy)  / (grid.slice(1, k + 1, grid.size(1)) - grid.slice(1, 1, grid.size(1) - k)) * bases.slice(2, 1, bases.size(2));
        bases = left_part + right_part;
    }

    assert(bases.sizes()[0] == x.sizes()[0]);
    assert(bases.sizes()[1] == in_features);
    assert(bases.sizes()[2] == grid_size + spline_order);

    return bases.contiguous();

    // return torch::zeros({x.size(0), out_features});

}


torch::Tensor KANLinear::curve2coeff(const torch::Tensor& x, const torch::Tensor& y) {
    assert(x.dim() == 2);
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

    // Prepare output tensors for lstsq_out
    torch::Tensor solution = torch::empty({A.size(0), A.size(2), B.size(2)}, A.options()); // Solution tensor
    torch::Tensor residuals = torch::empty({}, A.options());   // Residuals tensor
    torch::Tensor rank = torch::empty({}, A.options().dtype(torch::kInt));  // Rank tensor
    torch::Tensor singular_values = torch::empty({std::min(A.size(1), A.size(2))}, A.options()); // Singular values tensor

    torch::linalg_lstsq_out(solution, residuals, rank, singular_values, A, B, 1e-10);

    // auto lstsq_result = torch::linalg::lstsq(A, B, 1e-10, nullptr);
    // std::cout << "done lstsq"<< std::endl;
    // auto solution = std::get<0>(lstsq_result);  // Shape: [in_features, grid_size + spline_order, out_features]
    
    //using torch::linalg::pinv instead of torch::linalg::lstsq
    // torch::Tensor pseudo_inverse = torch::linalg::pinv(A);
    // torch::Tensor solution = torch::matmul(pseudo_inverse, B);

    // // Permute to get final shape
    torch::Tensor result = solution.permute({2, 0, 1});  // Shape: [out_features, in_features, grid_size + spline_order]
    
    assert(result.size(0) == out_features);
    assert(result.size(1) == in_features);
    assert(result.size(2) == grid_size + spline_order);
    
    return result.contiguous();
    // return torch::zeros({out_features, in_features, grid_size + spline_order});

}

torch::Tensor KANLinear::scaled_spline_weight() {
    if (enable_standalone_scale_spline) {
        return spline_weight * spline_scaler.unsqueeze(-1); // Broadcast the scaler across the weight tensor
    } 
    else {
        return spline_weight; // No scaling applied
    }
}

torch::Tensor KANLinear::forward(const torch::Tensor& x) {
    assert(x.size(-1) == in_features);
    auto original_shape = x.sizes();
    torch::Tensor x_copy = x.clone();
    x_copy = x_copy.view({-1, in_features});

    auto base_output = torch::linear(base_activation(x_copy), base_weight);

    auto spline_output = torch::linear(
    b_splines(x).view({x.size(0), -1}),
    scaled_spline_weight().view({out_features, -1})
    );

    auto output = base_output + spline_output;
    std::vector<int64_t> new_shape(original_shape.begin(), original_shape.end() - 1);  // Exclude the last dimension
    new_shape.push_back(out_features);  // Add the new dimension (out_features)
    output = output.view(new_shape);

    return output;

    // return torch::zeros({x.size(0), out_features});
}

void KANLinear::update_grid(const torch::Tensor& x, double margin) {
    torch::NoGradGuard no_grad;
    // std::cout << "x shape before assert: " << x.sizes() << std::endl;
    assert(x.dim() == 2);
    assert(x.size(1) == in_features);

    int64_t batch = x.size(0);

    // Compute splines using the b_splines method
    // std::cout << "x shape before b_splines: " << x.sizes() << std::endl;
    torch::Tensor splines = b_splines(x);  // shape: (batch, in, coeff)
    // std::cout << "x shape after b_splines: " << x.sizes() << std::endl;
    splines = splines.permute({1, 0, 2});  // shape: (in, batch, coeff)

    // Access original coefficients (scaled_spline_weight)
    torch::Tensor orig_coeff = scaled_spline_weight();  // shape: (out, in, coeff)
    orig_coeff = orig_coeff.permute({1, 2, 0});  // shape: (in, coeff, out)

    // Perform batch matrix multiplication
    torch::Tensor unreduced_spline_output = torch::bmm(splines, orig_coeff);  // shape: (in, batch, out)
    unreduced_spline_output = unreduced_spline_output.permute({1, 0, 2});  // shape: (batch, in, out)
    // std::cout << "unreduced_spline_output shape: " << unreduced_spline_output.sizes() << std::endl;

    // Sort input tensor x to get x_sorted
    torch::Tensor x_sorted = std::get<0>(x.sort(0)).squeeze(-1);  // sorting along dimension 0
    // std::cout << "x_sorted shape: " << x_sorted.sizes() << std::endl;
    // std::cout << "x_sorted[0] shape: " << x_sorted[0].sizes() << std::endl;
    torch::Tensor grid_adaptive = x_sorted.index({
        torch::linspace(0, batch - 1, grid_size + 1, torch::kInt64).to(x.device())
    });
    // std::cout << "done grid_adaptive" << std::endl;

    // Extract the scalar values from the tensors and compute the uniform step
    torch::Tensor uniform_step = ((x_sorted[-1] - x_sorted[0] + 2 * margin) / grid_size);
    // std::cout << "done uniform_step" << std::endl;

    // Create uniform grid
    torch::Tensor grid_uniform = torch::arange(grid_size + 1, torch::TensorOptions().dtype(torch::kFloat32).device(x.device())).unsqueeze(1);
    // std::cout << "grid_uniform shape: " << grid_uniform.sizes() << std::endl;
    // std::cout << "uniform_step shape: " << uniform_step.sizes() << std::endl;

    grid_uniform = (grid_uniform * uniform_step) + x_sorted[0]; // - margin;
    // std::cout << "grid_uniform(after * with uniform_step): " << grid_uniform.sizes() << std::endl;

    // Combine adaptive and uniform grids with the epsilon term
    torch::Tensor grid = grid_eps * grid_uniform + (1 - grid_eps) * grid_adaptive;

    // Create final grid by adding spline_order terms
    grid = torch::cat({
        grid.index({torch::indexing::Slice(0, 1)}) -
        uniform_step * torch::arange(spline_order, 0, -1, x.device()).unsqueeze(1),
        grid,
        grid.index({torch::indexing::Slice(-1)}) +
        uniform_step * torch::arange(1, spline_order + 1, x.device()).unsqueeze(1)
    }, 0);

    // Update the grid
    this->grid.copy_(grid.transpose(0, 1));

    // Update the spline weights using curve2coeff
    // std::cout << "x shape before curve2coeff: " << x.sizes() << std::endl;
    this->spline_weight.data().copy_(curve2coeff(x, unreduced_spline_output));
}

torch::Tensor KANLinear::regularization_loss(double regularize_activation, double regularize_entropy) {
    // Compute the L1 regularization term
    torch::Tensor l1_fake = this->spline_weight.abs().mean(-1);

    // Compute the activation regularization loss
    torch::Tensor regularization_loss_activation = l1_fake.sum();

    // Compute the probability distribution
    torch::Tensor p = l1_fake / regularization_loss_activation;

    // Compute the entropy regularization loss
    torch::Tensor regularization_loss_entropy = -torch::sum(p * p.log());

    // Combine and return the total regularization loss
    return (regularize_activation * regularization_loss_activation + regularize_entropy * regularization_loss_entropy);

    // return torch::tensor(0.0); // No-op, for now
}
