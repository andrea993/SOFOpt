#ifndef SOF_PROBLEM
#define SOF_PROBLEM

#include <Eigen/Dense>
#include <string>


#ifdef __cplusplus
extern "C" {
#endif
struct SOF_Problem
{
    Eigen::MatrixXd A;
    Eigen::MatrixXd B;
    Eigen::MatrixXd C;
    Eigen::MatrixXd Q;
    Eigen::MatrixXd R;
    Eigen::VectorXd X0;
    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> Structure;
    bool use_P_precond = true;
    double rho_alpha = 1e-6;
    double beta = 100.0; // Softmax spectral absicssa
    double r = 1e-5; //Softplus cost
    double c = 500; //Softplus cost
};

struct SOF_Result
{
    Eigen::MatrixXd Kx;
    Eigen::MatrixXd Ky;
    double init_cost;
    double optim_cost;
    std::string result;
};


#ifdef __cplusplus
}
#endif

#endif
