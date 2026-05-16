#include "SOFOpt_Core.hpp"

#include <cmath>
#include <stdexcept>

#include "SOFOpt_SQP.hpp"

SOF_Result SolveOutputFeedback(const SOF_Problem& problem)
{
    if (problem.A.rows() <= 0 || problem.A.cols() <= 0 ||
        problem.B.rows() <= 0 || problem.B.cols() <= 0 ||
        problem.C.rows() <= 0 || problem.C.cols() <= 0 ||
        problem.Q.rows() <= 0 || problem.Q.cols() <= 0 ||
        problem.R.rows() <= 0 || problem.R.cols() <= 0 ||
        problem.X0.size() <= 0 ||
        problem.Structure.rows() <= 0 || problem.Structure.cols() <= 0) {
        throw std::invalid_argument("Invalid matrix dimensions");
    }

    if (problem.A.rows() != problem.A.cols() ||
        problem.A.rows() != problem.B.rows() ||
        problem.C.cols() != problem.A.cols() ||
        problem.Q.rows() != problem.A.rows() ||
        problem.Q.cols() != problem.A.cols() ||
        problem.R.rows() != problem.B.cols() ||
        problem.R.cols() != problem.B.cols() ||
        problem.X0.size() != problem.A.rows() ||
        problem.Structure.rows() != problem.B.cols() ||
        problem.Structure.cols() != problem.C.rows()) {
        throw std::invalid_argument("Inconsistent matrix dimensions");
    }

    if (!std::isfinite(problem.rho_alpha) || problem.rho_alpha < 0.0) {
        throw std::invalid_argument("rho_alpha must be a non-negative finite scalar");
    }
    if (!std::isfinite(problem.beta) || problem.beta <= 0.0) {
        throw std::invalid_argument("beta must be a positive finite scalar");
    }
    if (!std::isfinite(problem.r) || problem.r < 0.0) {
        throw std::invalid_argument("r must be a non-negative finite scalar");
    }
    if (!std::isfinite(problem.c) || problem.c <= 0.0) {
        throw std::invalid_argument("c must be a positive finite scalar");
    }

    return SOFOpt_SQP::Optim_KxOut(problem);
}
