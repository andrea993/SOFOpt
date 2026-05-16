#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>

#include <Eigen/Eigenvalues>

#include "ControlAlgebra.hpp"
#include "SOFOpt_Core.hpp"

namespace
{
SOF_Problem MakeDebugProblem()
{
    SOF_Problem problem;

    problem.A.resize(4, 4);
    problem.A << -1.0, 0.0, 1.0, 0.0,
                 0.0, -1.0, 0.0, -1.0,
                 -1.0, 0.0, -1.0, -1.0,
                 0.0, 1.0, -1.0, -1.0;

    problem.B.resize(4, 2);
    problem.B << 0.0, 0.0,
                 0.0, 0.0,
                 1.0, 0.0,
                 0.0, 1.0;

    problem.C.resize(2, 4);
    problem.C << 1.0, 1.0, 0.0, 0.0,
                 0.0, 1.0, 0.0, 0.0;

    problem.Q = Eigen::MatrixXd::Identity(4, 4);
    problem.R = Eigen::MatrixXd::Identity(2, 2);
    problem.X0.resize(4);
    problem.X0 << 1.0, 1.0, 1.0, 1.0;

    problem.Structure.resize(2, 2);
    problem.Structure << true, false,
                         false, true;

    problem.use_P_precond = true;
    return problem;
}

void PrintUsage(const char* program_name)
{
    std::cerr
        << "Usage: " << program_name << " [--no-precond] [--rho-alpha value]\n"
        << "Runs a built-in output-feedback debug case and prints the solution.\n";
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
    SOF_Problem problem = MakeDebugProblem();
    if (!ParseArgs(argc, argv, problem)) {
        PrintUsage(argv[0]);
        return 2;
    }

    try {
        const Eigen::MatrixXd lqr_gain =
            ControlAlgebra::K_LQRc(problem.A, problem.B, problem.Q, problem.R).first;
        const SOF_Result solution = SolveOutputFeedback(problem);

        std::cout << "Debug case configuration\n";
        std::cout << "use_P_precond: " << std::boolalpha << problem.use_P_precond << "\n\n";
        std::cout << "rho_alpha: " << problem.rho_alpha << "\n\n";

        std::cout << "Kx (LQR):\n" << lqr_gain << "\n\n";
        std::cout << "Kx (optimized):\n" << solution.Kx << "\n\n";
        std::cout << "Ky (optimized):\n" << solution.Ky << "\n\n";
        std::cout << "init_cost: " << solution.init_cost << '\n';
        std::cout << "optim_cost: " << solution.optim_cost << '\n';
        std::cout << "result: " << solution.result << "\n\n";

        std::cout << "eig(A - B*Kx_lqr):\n" << (problem.A - problem.B * lqr_gain).eigenvalues() << "\n\n";
        std::cout << "eig(A - B*Kx_opt):\n" << (problem.A - problem.B * solution.Kx).eigenvalues() << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Debug executable failed: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
