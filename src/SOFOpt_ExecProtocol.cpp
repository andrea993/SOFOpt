#include "SOFOpt_ExecProtocol.hpp"

#include <algorithm>
#include <cstdint>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <vector>

namespace
{
template <typename T>
T ReadScalar(std::istream& input)
{
    T value{};
    input.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!input) {
        throw std::runtime_error("Unable to read solver input stream");
    }
    return value;
}

template <typename T>
void WriteScalar(std::ostream& output, const T& value)
{
    output.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!output) {
        throw std::runtime_error("Unable to write solver output stream");
    }
}

Eigen::MatrixXd ReadDoubleMatrix(std::istream& input, std::int32_t rows, std::int32_t cols)
{
    if (rows <= 0 || cols <= 0) {
        throw std::runtime_error("Invalid matrix size in solver input");
    }

    Eigen::MatrixXd matrix(rows, cols);
    input.read(reinterpret_cast<char*>(matrix.data()), static_cast<std::streamsize>(matrix.size() * sizeof(double)));
    if (!input) {
        throw std::runtime_error("Unable to read matrix payload from solver input");
    }
    return matrix;
}

Eigen::VectorXd ReadDoubleVector(std::istream& input, std::int32_t size)
{
    if (size <= 0) {
        throw std::runtime_error("Invalid vector size in solver input");
    }

    Eigen::VectorXd vector(size);
    input.read(reinterpret_cast<char*>(vector.data()), static_cast<std::streamsize>(vector.size() * sizeof(double)));
    if (!input) {
        throw std::runtime_error("Unable to read vector payload from solver input");
    }
    return vector;
}

Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> ReadLogicalMatrix(
    std::istream& input,
    std::int32_t rows,
    std::int32_t cols)
{
    if (rows <= 0 || cols <= 0) {
        throw std::runtime_error("Invalid logical matrix size in solver input");
    }

    std::vector<std::uint8_t> raw(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols));
    input.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size()));
    if (!input) {
        throw std::runtime_error("Unable to read logical matrix payload from solver input");
    }

    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> matrix(rows, cols);
    for (std::int32_t col = 0; col < cols; ++col) {
        for (std::int32_t row = 0; row < rows; ++row) {
            const std::size_t index = static_cast<std::size_t>(col) * static_cast<std::size_t>(rows) + static_cast<std::size_t>(row);
            matrix(row, col) = raw[index] != 0;
        }
    }
    return matrix;
}

void WriteDoubleMatrix(std::ostream& output, const Eigen::MatrixXd& matrix)
{
    if (matrix.size() == 0) {
        return;
    }

    output.write(
        reinterpret_cast<const char*>(matrix.data()),
        static_cast<std::streamsize>(matrix.size() * sizeof(double)));
    if (!output) {
        throw std::runtime_error("Unable to write matrix payload to solver output");
    }
}
}

namespace SOFOpt_ExecProtocol
{
SOF_Problem ReadProblem(std::istream& input)
{
    if (ReadScalar<std::uint32_t>(input) != kMagic) {
        throw std::runtime_error("Invalid solver request header");
    }

    const auto a_rows = ReadScalar<std::int32_t>(input);
    const auto a_cols = ReadScalar<std::int32_t>(input);
    const auto b_rows = ReadScalar<std::int32_t>(input);
    const auto b_cols = ReadScalar<std::int32_t>(input);
    const auto c_rows = ReadScalar<std::int32_t>(input);
    const auto c_cols = ReadScalar<std::int32_t>(input);
    const auto q_rows = ReadScalar<std::int32_t>(input);
    const auto q_cols = ReadScalar<std::int32_t>(input);
    const auto r_rows = ReadScalar<std::int32_t>(input);
    const auto r_cols = ReadScalar<std::int32_t>(input);
    const auto x0_size = ReadScalar<std::int32_t>(input);
    const auto structure_rows = ReadScalar<std::int32_t>(input);
    const auto structure_cols = ReadScalar<std::int32_t>(input);
    const auto use_p_precond = ReadScalar<std::uint8_t>(input);
    const auto rho_alpha = ReadScalar<double>(input);
    const auto beta = ReadScalar<double>(input);
    const auto r = ReadScalar<double>(input);
    const auto c = ReadScalar<double>(input);

    SOF_Problem problem;
    problem.A = ReadDoubleMatrix(input, a_rows, a_cols);
    problem.B = ReadDoubleMatrix(input, b_rows, b_cols);
    problem.C = ReadDoubleMatrix(input, c_rows, c_cols);
    problem.Q = ReadDoubleMatrix(input, q_rows, q_cols);
    problem.R = ReadDoubleMatrix(input, r_rows, r_cols);
    problem.X0 = ReadDoubleVector(input, x0_size);
    problem.Structure = ReadLogicalMatrix(input, structure_rows, structure_cols);
    problem.use_P_precond = use_p_precond != 0;
    problem.rho_alpha = rho_alpha;
    problem.beta = beta;
    problem.r = r;
    problem.c = c;
    return problem;
}

void WriteResponse(std::ostream& output, const SolverResponse& response)
{
    const auto& result = response.result;
    const std::int32_t kx_rows = response.status == kSuccess ? static_cast<std::int32_t>(result.Kx.rows()) : 0;
    const std::int32_t kx_cols = response.status == kSuccess ? static_cast<std::int32_t>(result.Kx.cols()) : 0;
    const std::int32_t ky_rows = response.status == kSuccess ? static_cast<std::int32_t>(result.Ky.rows()) : 0;
    const std::int32_t ky_cols = response.status == kSuccess ? static_cast<std::int32_t>(result.Ky.cols()) : 0;
    const std::string message = response.status == kSuccess
        ? result.result
        : (response.message.empty() ? result.result : response.message);
    const auto message_length = static_cast<std::int32_t>(std::min<std::size_t>(message.size(), 4096U));

    WriteScalar(output, kMagic);
    WriteScalar(output, response.status);
    WriteScalar(output, kx_rows);
    WriteScalar(output, kx_cols);
    WriteScalar(output, ky_rows);
    WriteScalar(output, ky_cols);
    WriteScalar(output, result.init_cost);
    WriteScalar(output, result.optim_cost);
    WriteScalar(output, message_length);
    if (message_length > 0) {
        output.write(message.data(), message_length);
        if (!output) {
            throw std::runtime_error("Unable to write message payload to solver output");
        }
    }

    if (response.status == kSuccess) {
        WriteDoubleMatrix(output, result.Kx);
        WriteDoubleMatrix(output, result.Ky);
    }
}
}
