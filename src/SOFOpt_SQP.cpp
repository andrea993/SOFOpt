#include "SOFOpt_SQP.hpp"

SOF_Result SOFOpt_SQP::Optim_KxOut(const SOF_Problem &problem)
{
    int Nx = problem.A.rows();
    int Nu = problem.B.cols();
    int Ny = problem.C.rows();

    std::vector<std::pair<int, int>> structureIndices;
    if ((problem.Structure.rows() > 0 || problem.Structure.cols() > 0))
    {
        if (problem.Structure.rows() != Nu || problem.Structure.cols() != Ny)
            throw std::invalid_argument("The structure matrix must have the same dimensions as the Ky matrix");

        for (int i = 0; i < problem.Structure.rows(); ++i)
        {
            for (int j = 0; j < problem.Structure.cols(); ++j)
            {
                if (!problem.Structure(i, j))
                    structureIndices.push_back(std::make_pair(i, j));
            }
        }
    }

    auto ss = ControlAlgebra::StateSpace{problem.A, problem.B, problem.C, Eigen::MatrixXd::Zero(problem.C.rows(), problem.B.cols())};

    auto LQR_sol = ControlAlgebra::K_LQRc(ss, problem.Q, problem.R);
    auto Kx = LQR_sol.first;
    auto P = LQR_sol.second;

    Eigen::MatrixXd Psqrt;
    if (problem.use_P_precond) // Change of coordinates (coordinates "P") to improve conditioning using the solution of the ARE as a preconditioner
    {
        Eigen::MatrixXd L = P.llt().matrixL();    
        Psqrt = L.transpose().triangularView<Eigen::Upper>().solve(Eigen::MatrixXd::Identity(Nx, Nx));// preconditioning matrix
    }
    else
    {
        Psqrt = MatrixXd::Identity(Nx, Nx);
    }

    ControlAlgebra::StateSpace ss_P = ControlAlgebra::StateTransform(ss, Psqrt.inverse());
    Eigen::MatrixXd Kx_P = Kx * Psqrt;
    Eigen::MatrixXd Q_P = Psqrt.transpose() * problem.Q * Psqrt;
    Eigen::MatrixXd R_P = problem.R;
    Eigen::VectorXd X0_P = Psqrt.inverse() * problem.X0;

    auto Acl_lqr = ss_P.A - ss_P.B*Kx_P;
    auto sp_pars = ControlAlgebra::SoftPlusParsFromPoles(Acl_lqr, problem.r, problem.c);

    const double s_alpha_max =
        ControlAlgebra::SoftSpectralAbscissaSchurValue(Acl_lqr, problem.beta) * problem.rho_alpha;
        

    double LQR_Cost = ControlAlgebra::LQR_CostEval(Kx_P, ss_P, Q_P, R_P, X0_P);

    auto OutputNullSpace = ControlAlgebra::Ker(ss_P.C);
    auto OutputUsefulSpace = ss_P.C.transpose() * (ss_P.C * ss_P.C.transpose()).inverse(); // pinv(C), complementary to ker(C)


    // Create an OptData structure to hold the parameters
    auto constant_NullGain_Gradient = Constant_NullGain_Gradient(ss, OutputNullSpace);
    auto constant_Structure_Gradient = Constant_Structure_Gradient(ss, OutputUsefulSpace, structureIndices);

    SOFOpt_SQP::OptData optData = {
        ss_P, Q_P, R_P, X0_P, LQR_Cost, OutputNullSpace, OutputUsefulSpace,
        structureIndices,
        constant_NullGain_Gradient,
        constant_Structure_Gradient,
        sp_pars,
        s_alpha_max,
        problem.beta
        };

    // NLOPT_LN_COBYLA NLOPT_LD_SLSQP
    nlopt_opt opt = nlopt_create(NLOPT_LD_SLSQP, Nx * Nu);

    // Null gain constraints
    auto nullGain_constraint = [](unsigned m, double *result, unsigned n, const double *x, double *grad, void *data) -> void
    {
        auto optData_ptr = reinterpret_cast<OptData *>(data);
        int Nx = optData_ptr->ss.A.rows();
        int Nu = optData_ptr->ss.B.cols();

        if (grad != nullptr)
        {
            std::copy(optData_ptr->constant_NullGain_Gradient.data(),
                      optData_ptr->constant_NullGain_Gradient.data() + optData_ptr->constant_NullGain_Gradient.size(),
                      grad);
        }

        auto Kx = Eigen::Map<const Eigen::MatrixXd>(x, Nu, Nx);
        auto K_null = Eigen::Map<Eigen::MatrixXd>(result, Nu, optData_ptr->OutputNullSpace.cols());
        K_null = Kx * optData_ptr->OutputNullSpace;
    };
    auto nullGain_constraint_TOL = std::vector<double>(Nu * OutputNullSpace.cols(), 1e-6);
    nlopt_add_equality_mconstraint(opt, Nu * OutputNullSpace.cols(), nullGain_constraint, &optData, nullGain_constraint_TOL.data());

    // Strucutre constraints
    auto structure_constraint = [](unsigned m, double *result, unsigned n, const double *x, double *grad, void *data) -> void
    {
        auto optData_ptr = reinterpret_cast<OptData *>(data);
        int Nx = optData_ptr->ss.A.rows();
        int Nu = optData_ptr->ss.B.cols();

        if (grad != nullptr)
        {
            std::copy(optData_ptr->constant_Structure_Gradient.data(),
                      optData_ptr->constant_Structure_Gradient.data() + optData_ptr->constant_Structure_Gradient.size(),
                      grad);
        }

        auto Kx = Eigen::Map<const Eigen::MatrixXd>(x, Nu, Nx);
        for (int i = 0; i < optData_ptr->structureIndices.size(); ++i)
        {
            const auto &s_idx = optData_ptr->structureIndices[i];
            result[i] = Kx.row(s_idx.first) * optData_ptr->OutputUsefulSpace.col(s_idx.second);
        }
    };
    auto structure_constraint_TOL = std::vector<double>(structureIndices.size(), 1e-6);
    nlopt_add_equality_mconstraint(opt, structureIndices.size(), structure_constraint, &optData, structure_constraint_TOL.data());

    // Schur spectral constraints
    auto eigen_constraint = [](unsigned n, const double *x, double *grad, void *data) -> double
    {
        auto optData_ptr = reinterpret_cast<OptData *>(data);

        int Nx = optData_ptr->ss.A.rows();
        int Nu = optData_ptr->ss.B.cols();

        auto Kx = Eigen::Map<const Eigen::MatrixXd>(x, Nu, Nx);
        auto Acl = optData_ptr->ss.A - optData_ptr->ss.B * Kx;
       
        if (grad != nullptr)
        {           
            auto [alpha, dAcldalpha] =
                ControlAlgebra::SoftSpectralAbscissaSchurGradient(Acl, optData_ptr->beta);
            Eigen::Map<Eigen::MatrixXd>(grad, Nu, Nx) = -optData_ptr->ss.B.transpose()*dAcldalpha;
            return alpha - optData_ptr->s_alpha_max;
        }
        else
        {
            return ControlAlgebra::SoftSpectralAbscissaSchurValue(Acl, optData_ptr->beta) - optData_ptr->s_alpha_max;
        }
    };
    auto eigen_constraint_TOL = 1e-6;
    nlopt_add_inequality_constraint(opt, eigen_constraint, &optData, eigen_constraint_TOL);


    nlopt_set_min_objective(opt, SOFOpt_SQP::Fitness, &optData);
    nlopt_set_xtol_rel(opt, 1e-8);
    //nlopt_set_maxeval(opt, 2048);

    double minf;
    auto result = nlopt_optimize(opt, Kx_P.data(), &minf);

    // Destroy the nlopt_opt object to free resources
    nlopt_destroy(opt);

    //convert result to original coordinates
    Kx = Kx_P * Psqrt.inverse();

    // std::cout << "ERROR MSG " << nlopt_get_errmsg(opt) << std::endl;

    //auto Kx_optim = Eigen::Map<MatrixXd>(x, Nu, Nx);
    auto Ky_optim = Kx_P * OutputUsefulSpace;
    return SOF_Result{Kx, Ky_optim, LQR_Cost, ControlAlgebra::LQR_CostEval(Kx, ss, problem.Q, problem.R, problem.X0), std::string(nlopt_result_to_string(result))};
}

double SOFOpt_SQP::Fitness(unsigned n, const double *x, double *grad, void *data)
{
    OptData *optData = static_cast<OptData *>(data);

    int Nx = optData->ss.A.rows();
    int Nu = optData->ss.B.cols();

    Eigen::Map<const Eigen::MatrixXd> Kx(x, Nu, Nx);

    double J;
    if (grad != nullptr)
    {
        auto res = ControlAlgebra::LQR_SoftCostEval<true>(
            Kx, optData->ss, optData->Q, optData->R, optData->X0, optData->sp_pars);

        J = res.first;
        Eigen::Map<Eigen::MatrixXd>(grad, Nu, Nx) = res.second;
    }
    else
    {
        J = ControlAlgebra::LQR_SoftCostEval<false>(
            Kx, optData->ss, optData->Q, optData->R, optData->X0, optData->sp_pars);
    }

    return J - optData->LQR_cost;
}

std::vector<double> SOFOpt_SQP::Constant_NullGain_Gradient(const ControlAlgebra::StateSpace &ss, const Eigen::MatrixXd &OutputNullSpace)
{
    int Nx = ss.A.rows();
    int Nu = ss.B.cols();
    int Nc = Nu * OutputNullSpace.cols();

    std::vector<double> grad(Nx * Nc * Nu);
    auto JacobMap = Eigen::Map<Eigen::MatrixXd>(grad.data(), Nu * Nx, Nc);

    for (int i = 0; i < Nu; ++i)
    {
        for (int j = 0; j < OutputNullSpace.cols(); ++j)
        {
            Eigen::MatrixXd gradMap = Eigen::MatrixXd::Zero(Nu, Nx);
            gradMap.row(i) = OutputNullSpace.col(j);
            JacobMap.col(j * Nu + i) = Eigen::Map<Eigen::VectorXd>(gradMap.data(), Nu * Nx);
        }
    }

    return grad;
}

std::vector<double> SOFOpt_SQP::Constant_Structure_Gradient(const ControlAlgebra::StateSpace &ss, const Eigen::MatrixXd &OutputUsefulSpace, const std::vector<std::pair<int, int>> &structureIndices)
{
    int Nx = ss.A.rows();
    int Nu = ss.B.cols();

    std::vector<double> grad(Nx * Nu * structureIndices.size());
    auto JacobMap = Eigen::Map<Eigen::MatrixXd>(grad.data(), Nu * Nx, structureIndices.size());

    for (int i = 0; i < structureIndices.size(); ++i)
    {
        const auto &s_idx = structureIndices[i];
        Eigen::MatrixXd gradMap = Eigen::MatrixXd::Zero(Nu, Nx);
        gradMap.row(s_idx.first) = OutputUsefulSpace.col(s_idx.second);
        JacobMap.col(i) = Eigen::Map<Eigen::VectorXd>(gradMap.data(), Nu * Nx);
    }

    return grad;
}
