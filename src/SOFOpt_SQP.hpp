#pragma once

#include <vector>
#include <iostream>
#include <memory>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include "ControlAlgebra.hpp"
#include "SOFOpt_Core.hpp"
#include "SOF_Problem.h"

#include <nlopt.h>


class SOFOpt_SQP
{
public:
 
    struct OptData
    {
        ControlAlgebra::StateSpace ss;
        Eigen::MatrixXd Q;
        Eigen::MatrixXd R;
        Eigen::VectorXd X0;
        double LQR_cost;
        Eigen::MatrixXd OutputNullSpace;
        Eigen::MatrixXd OutputUsefulSpace;
        std::vector<std::pair<int, int>> structureIndices;
        std::vector<double> constant_NullGain_Gradient;
        std::vector<double> constant_Structure_Gradient;
        ControlAlgebra::SoftPlus_Pars sp_pars;
        double s_alpha_max;
        double beta = 100.0; // Softmax spectral absicssa
    };

    static SOF_Result Optim_KxOut(const SOF_Problem& problem);

    static double Fitness(unsigned n, const double *x, double *grad, void *data);

    static std::vector<double> Constant_NullGain_Gradient(const ControlAlgebra::StateSpace &ss, const Eigen::MatrixXd &OutputNullSpace);
    static std::vector<double> Constant_Structure_Gradient(const ControlAlgebra::StateSpace &ss, const Eigen::MatrixXd &OutputUsefulSpace, const std::vector<std::pair<int, int>> &structureIndices);

};