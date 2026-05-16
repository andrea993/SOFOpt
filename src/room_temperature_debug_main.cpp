#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>

#include <Eigen/Eigenvalues>

#include "ControlAlgebra.hpp"
#include "SOFOpt_Core.hpp"

namespace
{
SOF_Problem MakeRoomTemperatureProblem()
{
    constexpr int n_states = 10;
    constexpr int n_inputs = 3;
    constexpr int n_outputs = 4;

    SOF_Problem problem;
    problem.A = Eigen::MatrixXd::Zero(n_states, n_states);
    problem.B = Eigen::MatrixXd::Zero(n_states, n_inputs);
    problem.C = Eigen::MatrixXd::Zero(n_outputs, n_states);

    for (int i = 0; i < n_states; ++i) {
        problem.A(i, i) = -0.2;

        if (i < n_states - 1) {
            problem.A(i, i + 1) = 0.15;
            problem.A(i + 1, i) = 0.15;
        }

        if (i < n_states - 2) {
            problem.A(i, i + 2) = 0.05;
            problem.A(i + 2, i) = 0.05;
        }
    }

    problem.B.block<4, 1>(0, 0) << 0.8, 0.6, 0.4, 0.2;
    problem.B.block<4, 1>(3, 1) << 0.8, 0.6, 0.4, 0.2;
    problem.B.block<4, 1>(6, 2) << 0.8, 0.6, 0.4, 0.2;

    problem.C(0, 0) = 1.0;
    problem.C(0, 1) = 0.5;
    problem.C(0, 2) = 0.25;
    problem.C(1, 3) = 1.0;
    problem.C(1, 4) = 0.5;
    problem.C(1, 5) = 0.25;
    problem.C(2, 6) = 1.0;
    problem.C(2, 7) = 0.5;
    problem.C(2, 8) = 0.25;
    problem.C(3, 2) = 0.25;
    problem.C(3, 5) = 0.25;
    problem.C(3, 8) = 0.25;
    problem.C(3, 9) = 1.0;

    problem.Q = Eigen::MatrixXd::Identity(n_states, n_states);
    problem.R = Eigen::MatrixXd::Identity(n_inputs, n_inputs);
    problem.X0 = Eigen::VectorXd::Ones(n_states);
    problem.Structure =
        Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>::Constant(n_inputs, n_outputs, true);
    problem.use_P_precond = true;
    problem.rho_alpha = 0.5;
    return problem;
}

double SpectralAlpha(const Eigen::MatrixXd& A)
{
    return A.eigenvalues().real().maxCoeff();
}

std::pair<double, Eigen::MatrixXd> SoftSpectralAbscissaEigen(const Eigen::MatrixXd& A, const double beta = 50.0)
{
    Eigen::EigenSolver<Eigen::MatrixXd> es;
    es.compute(A, true);
    if (es.info() != Eigen::Success) {
        throw std::runtime_error("EigenSolver failed in debug soft spectral check");
    }

    auto eigvals = es.eigenvalues().real();
    const double max_val = eigvals.maxCoeff();
    Eigen::VectorXd exp_vals = (beta * (eigvals.array() - max_val)).exp();
    const double sum_exp = exp_vals.sum();
    const double alpha = max_val + std::log(sum_exp) / beta;

    const auto& V = es.eigenvectors();
    const Eigen::VectorXd w = exp_vals / sum_exp;
    const Eigen::MatrixXcd D = w.cast<std::complex<double>>().asDiagonal();
    const Eigen::MatrixXd gradient = (V * D * V.inverse()).real();

    return {alpha, gradient};
}

void PrintUsage(const char* program_name)
{
    std::cerr
        << "Usage: " << program_name << " [--no-precond] [--rho-alpha value]\n"
        << "Runs the built-in room temperature debug case and prints solver diagnostics.\n";
}

bool ParseArgs(int argc, char** argv, SOF_Problem& problem)
{
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];

        if (arg == "--no-precond") {
            problem.use_P_precond = false;
            continue;
        }

        if (arg == "--rho-alpha") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value after --rho-alpha\n";
                return false;
            }

            problem.rho_alpha = std::stod(argv[++i]);
            continue;
        }

        if (arg == "--help" || arg == "-h") {
            PrintUsage(argv[0]);
            std::exit(0);
        }

        std::cerr << "Unknown argument: " << arg << '\n';
        return false;
    }

    return true;
}
} // namespace

int main(int argc, char** argv)
{
    SOF_Problem problem = MakeRoomTemperatureProblem();
    if (!ParseArgs(argc, argv, problem)) {
        PrintUsage(argv[0]);
        return 2;
    }

    try {
        const auto lqr = ControlAlgebra::K_LQRc(problem.A, problem.B, problem.Q, problem.R);
        const Eigen::MatrixXd& Kx_lqr = lqr.first;
        const double alpha_lqr = SpectralAlpha(problem.A - problem.B * Kx_lqr);
        const Eigen::MatrixXd output_useful =
            problem.C.transpose() * (problem.C * problem.C.transpose()).inverse();
        const Eigen::MatrixXd ky_projected = Kx_lqr * output_useful;
        const Eigen::MatrixXd kx_projected = ky_projected * problem.C;
        const double alpha_projected = SpectralAlpha(problem.A - problem.B * kx_projected);
        std::cout << "pre-solve alpha_lqr Eigen: " << alpha_lqr << '\n';
        std::cout << "pre-solve alpha projected output LQR Eigen: " << alpha_projected << '\n';
        std::cout << "pre-solve alpha_lqr Schur: "
                  << ControlAlgebra::SpectralAbscissaSchurValue(problem.A - problem.B * Kx_lqr)
                  << '\n';
        std::cout << "pre-solve soft alpha_lqr Schur: "
                  << ControlAlgebra::SoftSpectralAbscissaSchurValue(problem.A - problem.B * Kx_lqr, 50.0)
                  << '\n';
        const auto [soft_alpha_eigen, soft_grad_eigen] =
            SoftSpectralAbscissaEigen(problem.A - problem.B * Kx_lqr);
        const auto [soft_alpha_schur, soft_grad_schur] =
            ControlAlgebra::SoftSpectralAbscissaSchurGradient(problem.A - problem.B * Kx_lqr, 50.0);
        std::cout << "pre-solve soft alpha Eigen gradient path: " << soft_alpha_eigen << '\n';
        std::cout << "pre-solve soft alpha Schur gradient path: " << soft_alpha_schur << '\n';
        std::cout << "pre-solve soft gradient max abs diff: "
                  << (soft_grad_schur - soft_grad_eigen).cwiseAbs().maxCoeff() << "\n\n";
        const double alpha_bound = problem.rho_alpha * alpha_lqr;

        const SOF_Result solution = SolveOutputFeedback(problem);
        const double alpha_kx = SpectralAlpha(problem.A - problem.B * solution.Kx);
        const double alpha_ky = SpectralAlpha(problem.A - problem.B * solution.Ky * problem.C);

        std::cout << "Room temperature debug case\n";
        std::cout << "use_P_precond: " << std::boolalpha << problem.use_P_precond << '\n';
        std::cout << "rho_alpha: " << problem.rho_alpha << "\n\n";

        std::cout << "Kx (LQR):\n" << Kx_lqr << "\n\n";
        std::cout << "Kx (optimized):\n" << solution.Kx << "\n\n";
        std::cout << "Ky (optimized):\n" << solution.Ky << "\n\n";
        std::cout << "init_cost: " << solution.init_cost << '\n';
        std::cout << "optim_cost: " << solution.optim_cost << '\n';
        std::cout << "result: " << solution.result << "\n\n";
        std::cout << "alpha_lqr: " << alpha_lqr << '\n';
        std::cout << "alpha_bound: " << alpha_bound << '\n';
        std::cout << "alpha(A - B*Kx_opt): " << alpha_kx << '\n';
        std::cout << "alpha(A - B*Ky*C): " << alpha_ky << '\n';
        std::cout << "bound residual Ky: " << (alpha_ky - alpha_bound) << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Room temperature debug executable failed: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
