#include "SOF_Problem.h"
#include "SOFOpt_SQP.hpp"
#include "SOFOpt_LibItf.h"
#include <Eigen/Dense>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <exception>
#include <cmath>

//#define DEBUG

OptimOutput_C Optim_KxOut_C(
    const double* A, int A_rows, int A_cols,
    const double* B, int B_rows, int B_cols,
    const double* C, int C_rows, int C_cols,
    const double* Q, int Q_rows, int Q_cols,
    const double* R, int R_rows, int R_cols,
    const double* X0, int X0_size,
    const bool* Structure, int Structure_rows, int Structure_cols,
    bool use_P_precond,
    double rho_alpha, double beta, double r, double c
)
{
    OptimOutput_C result{};
    result.Kx = nullptr;
    result.Ky = nullptr;
    result.init_cost = 0.0;
    result.optim_cost = 0.0;
    std::strcpy(result.result, "Unknown optimization error");

    try {

#ifdef DEBUG
    std::cout << "Inside Optim_KxOutn" << std::endl;
#endif

        // Input validation
    if (!A || !B || !C || !Q || !R || !X0 || !Structure) {
        throw std::invalid_argument("Null pointer passed as argument");
    }

    if (A_rows <= 0 || A_cols <= 0 || B_rows <= 0 || B_cols <= 0 ||
        C_rows <= 0 || C_cols <= 0 || Q_rows <= 0 || Q_cols <= 0 ||
        R_rows <= 0 || R_cols <= 0 || X0_size <= 0 ||
        Structure_rows <= 0 || Structure_cols <= 0) {
        throw std::invalid_argument("Invalid matrix dimensions");
    }

    // Dimension consistency checks
    if (A_rows != A_cols || A_rows != B_rows || 
        A_cols != Q_rows || Q_rows != Q_cols ||
        B_cols != R_cols || R_rows != R_cols ||
        X0_size != A_rows) {
        throw std::invalid_argument("Inconsistent matrix dimensions");
    }

    if (!std::isfinite(rho_alpha) || rho_alpha < 0.0) {
        throw std::invalid_argument("rho_alpha must be a non-negative finite scalar");
    }
    if (!std::isfinite(beta) || beta <= 0.0) {
        throw std::invalid_argument("beta must be a positive finite scalar");
    }
    if (!std::isfinite(r) || r < 0.0) {
        throw std::invalid_argument("r must be a non-negative finite scalar");
    }
    if (!std::isfinite(c) || c <= 0.0) {
        throw std::invalid_argument("c must be a positive finite scalar");
    }
    

#ifdef DEBUG
    std::cout << "A_rows: " << A_rows << ", A_cols: " << A_cols << std::endl;
    std::cout << "B_rows: " << B_rows << ", B_cols: " << B_cols << std::endl;
    std::cout << "C_rows: " << C_rows << ", C_cols: " << C_cols << std::endl;
    std::cout << "Q_rows: " << Q_rows << ", Q_cols: " << Q_cols << std::endl;
    std::cout << "R_rows: " << R_rows << ", R_cols: " << R_cols << std::endl;
    std::cout << "X0_size: " << X0_size << std::endl;
    std::cout << "Structure_rows: " << Structure_rows << ", Structure_cols: " << Structure_cols << std::endl;
#endif


    Eigen::Map<const Eigen::MatrixXd> A_map(A, A_rows, A_cols);
    Eigen::Map<const Eigen::MatrixXd> B_map(B, B_rows, B_cols);
    Eigen::Map<const Eigen::MatrixXd> C_map(C, C_rows, C_cols);
    Eigen::Map<const Eigen::MatrixXd> Q_map(Q, Q_rows, Q_cols);
    Eigen::Map<const Eigen::MatrixXd> R_map(R, R_rows, R_cols);
    Eigen::Map<const Eigen::VectorXd> X0_map(X0, X0_size);
    Eigen::Map<const Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>> Structure_map(Structure, Structure_rows, Structure_cols);

#ifdef DEBUG
    std::cout << "A: \n" << A_map << std::endl;
    std::cout << "B: \n" << B_map << std::endl;
    std::cout << "C: \n" << C_map << std::endl;
    std::cout << "Q: \n" << Q_map << std::endl;
    std::cout << "R: \n" << R_map << std::endl;
    std::cout << "X0: \n" << X0_map << std::endl;
    std::cout << "Structure: \n" << Structure_map << std::endl;
#endif

    auto problem = SOF_Problem {
        A_map, B_map, C_map, Q_map, R_map, X0_map, Structure_map, use_P_precond, rho_alpha, beta, r, c
    };

    auto sol = SOFOpt_SQP::Optim_KxOut(problem);
       

#ifdef DEBUG
    std::cout << "Kx: \n" << sol.Kx << std::endl;
    std::cout << "Ky: \n" << sol.Ky << std::endl;
#endif

    result.Kx = static_cast<double*>(std::malloc(sol.Kx.size() * sizeof(double)));
    result.Ky = static_cast<double*>(std::malloc(sol.Ky.size() * sizeof(double)));
    if (!result.Kx || !result.Ky) {
        std::free(result.Kx);
        std::free(result.Ky);
        result.Kx = nullptr;
        result.Ky = nullptr;
        throw std::bad_alloc();
    }
    std::copy(sol.Kx.data(), sol.Kx.data() + sol.Kx.size(), result.Kx);
    std::copy(sol.Ky.data(), sol.Ky.data() + sol.Ky.size(), result.Ky);
    result.init_cost = sol.init_cost;
    result.optim_cost = sol.optim_cost;
    std::strncpy(result.result, sol.result.c_str(), sizeof(result.result) - 1);
    result.result[sizeof(result.result) - 1] = '\0';

    } catch (const std::exception& e) {
        std::free(result.Kx);
        std::free(result.Ky);
        result.Kx = nullptr;
        result.Ky = nullptr;
        std::strncpy(result.result, e.what(), sizeof(result.result) - 1);
        result.result[sizeof(result.result) - 1] = '\0';
    } catch (...) {
        std::free(result.Kx);
        std::free(result.Ky);
        result.Kx = nullptr;
        result.Ky = nullptr;
        std::strncpy(result.result, "Unknown non-standard exception", sizeof(result.result) - 1);
        result.result[sizeof(result.result) - 1] = '\0';
    }

    return result;
}
