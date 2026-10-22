#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

#include "SOFOpt_Core.hpp"
#include "SOFOpt_ExecProtocol.hpp"
#include "SOFOpt_LibItf.h"

namespace
{
SOF_Problem MakeScalarProblem()
{
    SOF_Problem problem;
    problem.A = Eigen::MatrixXd::Constant(1, 1, 1.0);
    problem.B = Eigen::MatrixXd::Constant(1, 1, 1.0);
    problem.C = Eigen::MatrixXd::Constant(1, 1, 1.0);
    problem.Q = Eigen::MatrixXd::Constant(1, 1, 1.0);
    problem.R = Eigen::MatrixXd::Constant(1, 1, 1.0);
    problem.X0 = Eigen::VectorXd::Constant(1, 1.0);
    problem.ConstraintGain =
        Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>::Constant(1, 1, false);
    problem.ConstraintGainValues = Eigen::MatrixXd::Zero(1, 1);
    problem.use_P_precond = false;
    return problem;
}

void Require(bool condition, const std::string& message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename T>
void WriteScalar(std::ostream& output, const T& value)
{
    output.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void TestCppApi()
{
    auto problem = MakeScalarProblem();
    const auto default_result = SolveOutputFeedback(problem);
    Require(std::abs(default_result.init_cost - (1.0 + std::sqrt(2.0))) < 1e-9,
            "Empty Kx0 no longer selects the LQR initialization");

    // Kx0 = 0 leaves A - B*Kx0 = 1 unstable; it must still be accepted.
    problem.Kx0 = Eigen::MatrixXd::Zero(1, 1);

    const auto result = SolveOutputFeedback(problem);
    Require(std::abs(result.init_cost - default_result.init_cost) < 1e-9,
            "Kx0 unexpectedly changed the LQR reference cost");

    problem.use_P_precond = true;
    const auto preconditioned_result = SolveOutputFeedback(problem);
    Require(std::abs(preconditioned_result.init_cost - default_result.init_cost) < 1e-9,
            "Custom Kx0 preconditioning changed the LQR reference cost");

    problem.Kx0 = Eigen::MatrixXd::Zero(2, 1);
    try {
        (void)SolveOutputFeedback(problem);
        throw std::runtime_error("Invalid Kx0 dimensions were accepted");
    } catch (const std::invalid_argument&) {
    }

    problem.Kx0 = Eigen::MatrixXd::Constant(
        1, 1, std::numeric_limits<double>::quiet_NaN());
    try {
        (void)SolveOutputFeedback(problem);
        throw std::runtime_error("A non-finite Kx0 was accepted");
    } catch (const std::invalid_argument&) {
    }
}

void TestCAbi()
{
    const double a = 1.0;
    const double b = 1.0;
    const double c_matrix = 1.0;
    const double q = 1.0;
    const double r_matrix = 1.0;
    const double x0 = 1.0;
    const double kx0 = 0.0;
    const bool constraint_gain = false;
    const double constraint_value = 0.0;

    OptimOutput_C result = Optim_KxOut_WithKx0_C(
        &a, 1, 1, &b, 1, 1, &c_matrix, 1, 1,
        &q, 1, 1, &r_matrix, 1, 1, &x0, 1,
        &kx0, 1, 1, &constraint_gain, 1, 1, &constraint_value, 1, 1,
        false, 1e-6, 100.0, 1e-5, 500.0);

    Require(result.Kx != nullptr && result.Ky != nullptr,
            std::string("The C ABI returned an error: ") + result.result);
    Require(std::abs(result.init_cost - (1.0 + std::sqrt(2.0))) < 1e-9,
            "The C ABI changed the LQR reference cost");
    std::free(result.Kx);
    std::free(result.Ky);
}

void TestBinaryProtocolV2()
{
    std::ostringstream output(std::ios::binary);
    const std::int32_t one = 1;
    const std::uint8_t no_precond = 0;
    const double rho_alpha = 1e-6;
    const double beta = 100.0;
    const double softplus_r = 1e-5;
    const double softplus_c = 500.0;
    const double matrices[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    const std::uint8_t constraint = 0;
    const double constraint_value = 0.0;
    const double kx0 = 0.0;

    WriteScalar(output, SOFOpt_ExecProtocol::kMagicWithKx0);
    for (int i = 0; i < 15; ++i) {
        WriteScalar(output, one);
    }
    WriteScalar(output, no_precond);
    WriteScalar(output, rho_alpha);
    WriteScalar(output, beta);
    WriteScalar(output, softplus_r);
    WriteScalar(output, softplus_c);
    output.write(reinterpret_cast<const char*>(matrices), sizeof(matrices));
    output.write(reinterpret_cast<const char*>(&constraint), sizeof(constraint));
    output.write(reinterpret_cast<const char*>(&constraint_value), sizeof(constraint_value));
    output.write(reinterpret_cast<const char*>(&kx0), sizeof(kx0));

    std::istringstream input(output.str(), std::ios::binary);
    const auto problem = SOFOpt_ExecProtocol::ReadProblem(input);
    Require(problem.Kx0.rows() == 1 && problem.Kx0.cols() == 1 &&
                problem.Kx0(0, 0) == 0.0,
            "The v2 executable protocol did not deserialize Kx0");
}
} // namespace

int main()
{
    try {
        TestCppApi();
        TestCAbi();
        TestBinaryProtocolV2();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
