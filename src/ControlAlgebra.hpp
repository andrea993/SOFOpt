#pragma once

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

using namespace Eigen;

class ControlAlgebra
{
public:
    struct StateSpace
    {
        MatrixXd A;
        MatrixXd B;
        MatrixXd C;
        MatrixXd D;
    };

    struct SoftPlus_Pars
    {
        double eta;
        double eps;
    };

    static std::pair<MatrixXcd, MatrixXcd> SortEig(const MatrixXd &A);

    static MatrixXd AREc(const MatrixXd &A, const MatrixXd &B, const MatrixXd &C);
    static MatrixXd AREcEigen(const MatrixXd &A, const MatrixXd &B, const MatrixXd &C);

    static MatrixXd SlyapEigen(const MatrixXd &A, const MatrixXd &Q);

    static MatrixXd Slyap(const MatrixXd &A, const MatrixXd &Q);

    static double SpectralAbscissaSchurValue(const Eigen::MatrixXd &A);
    static std::pair<double, Eigen::MatrixXd> SpectralAbscissaSchurGradient(const Eigen::MatrixXd &A);
    static double SoftSpectralAbscissaSchurValue(const Eigen::MatrixXd &A, double beta);
    static std::pair<double, Eigen::MatrixXd> SoftSpectralAbscissaSchurGradient(const Eigen::MatrixXd &A, double beta);

    static std::pair<MatrixXd, MatrixXd> K_LQRc(const MatrixXd &A, const MatrixXd &B, const MatrixXd &Qx, const MatrixXd &Qu);

    static std::pair<MatrixXd, MatrixXd> K_LQRc(const StateSpace &ss, const MatrixXd &Qx, const MatrixXd &Qu)
    {
        return K_LQRc(ss.A, ss.B, Qx, Qu);
    }

    static std::pair<MatrixXd, MatrixXd> K_LQRc(const MatrixXd &A, const MatrixXd &B, const MatrixXd &Qx, const MatrixXd &Qu, const MatrixXd &Qxu);

    static std::pair<MatrixXd, MatrixXd> K_LQRc(const StateSpace &ss, const MatrixXd &Qx, const MatrixXd &Qu, const MatrixXd &Qxu)
    {
        return K_LQRc(ss.A, ss.B, Qx, Qu, Qxu);
    }

    static MatrixXd Ker(const MatrixXd &X, double tol = 1e-6)
    {
        auto svd = X.jacobiSvd<Eigen::ComputeFullV>();

        auto S = svd.singularValues();
        auto V = svd.matrixV();

        auto rank = (S.array() > tol).count();

        auto ker = V(seq(0, V.rows() - 1), seq(rank, V.cols() - 1)).eval();
        return ker;
    }

    static StateSpace StateTransform(const StateSpace &ss, const MatrixXd &T)
    {
        return StateSpace{T * ss.A * T.inverse(), T * ss.B, ss.C * T.inverse(), ss.D};
    }

    static std::pair<StateSpace, MatrixXd> NullOutputTransform(StateSpace &ss)
    {
        auto ker = Ker(ss.C);
        auto T = MatrixXd::Zero(ss.A.rows(), ss.A.cols()).eval();
        T << ker.transpose(),
            ss.C;

        return std::make_pair(StateTransform(ss, T), T);
    }

    template <bool evalGradient = false>
    static auto LQR_CostEval(const Eigen::MatrixXd &Kx, const StateSpace &ss, const Eigen::MatrixXd &Q, const Eigen::MatrixXd &R, const Eigen::VectorXd &X0)
    {
        Eigen::MatrixXd Acl = ss.A - ss.B * Kx;
        Eigen::MatrixXd Qlyap = Q + Kx.transpose() * R * Kx;

        Eigen::MatrixXd P = Slyap(Acl.transpose(), Qlyap);

        double J = (X0.transpose() * P * X0)(0, 0);

        if constexpr (!evalGradient)
            return J;
        else
        {
            Eigen::MatrixXd S = ControlAlgebra::Slyap(Acl, X0 * X0.transpose());
            Eigen::MatrixXd G_J = ((R * Kx * S - ss.B.transpose() * P * S) * 2.0).eval();
            return std::pair{J, G_J};
        }
    }

    template <bool evalGradient = false>
    static auto LQR_SoftCostEval(
        const Eigen::MatrixXd &Kx,
        const StateSpace &ss,
        const Eigen::MatrixXd &Q,
        const Eigen::MatrixXd &R,
        const Eigen::VectorXd &X0,
        const SoftPlus_Pars &sp)
    {
        Eigen::MatrixXd Acl_raw = ss.A - ss.B * Kx;
        Eigen::MatrixXd Qlyap = Q + Kx.transpose() * R * Kx;

        if constexpr (!evalGradient)
        {
            auto alpha = SoftSpectralAbscissa<false>(Acl_raw);
            auto gamma = SoftPlus_eta<false>(alpha, sp);

            Eigen::MatrixXd Acl =
                Acl_raw - gamma * Eigen::MatrixXd::Identity(Acl_raw.rows(), Acl_raw.cols());

            Eigen::MatrixXd P = Slyap(Acl.transpose(), Qlyap);

            double J = (X0.transpose() * P * X0)(0, 0);
            return J;
        }
        else
        {
            auto [alpha, G_alpha] = SoftSpectralAbscissa<true>(Acl_raw);
            auto [gamma, dgamma_dalpha] = SoftPlus_eta<true>(alpha, sp);

            Eigen::MatrixXd Acl =
                Acl_raw - gamma * Eigen::MatrixXd::Identity(Acl_raw.rows(), Acl_raw.cols());

            Eigen::MatrixXd P = Slyap(Acl.transpose(), Qlyap);

            double J = (X0.transpose() * P * X0)(0, 0);

            Eigen::MatrixXd S = Slyap(Acl, X0 * X0.transpose());

            Eigen::MatrixXd G_J =
                2.0 * (R * Kx * S - ss.B.transpose() * P * S);

            double dJ_dgamma =
                -2.0 * (P * S).trace();

            Eigen::MatrixXd dgamma_dK =
                -dgamma_dalpha * ss.B.transpose() * G_alpha;

            G_J += dJ_dgamma * dgamma_dK;

            return std::pair{J, G_J};
        }
    }

    template <bool evalGradient = false>
    static auto SpectralAbscissa(const Eigen::MatrixXd &A)
    {
        if constexpr (evalGradient)
        {
            return SpectralAbscissaSchurGradient(A);
        }
        else
        {
            return SpectralAbscissaSchurValue(A);
        }
    }

    template <bool evalGradient = false>
    static auto SoftSpectralAbscissa(const Eigen::MatrixXd &A, double beta = 50.0)
    {
        if constexpr (evalGradient)
        {
            return SoftSpectralAbscissaSchurGradient(A, beta);
        }
        else
        {
            return SoftSpectralAbscissaSchurValue(A, beta);
        }
    }

    static SoftPlus_Pars SoftPlusParsFromPoles(const Eigen::MatrixXd &A, double r = 1e-5, double c = 500)
    {
        auto alpha = SpectralAbscissaSchurValue(A);
        if (alpha >= 0)
            throw std::logic_error("Cannot evaluate soft plus parameters for unstable plant");

        return SoftPlus_Pars{-c / alpha, -r * alpha};
    }

    template <bool evalGradient = false>
    static auto SoftPlus_eta(double x, const SoftPlus_Pars &sp)
    {
        const double z = sp.eta * (x + sp.eps);
        const double s = (z > 0.0)
            ? (z + std::log1p(std::exp(-z))) / sp.eta
            : std::log1p(std::exp(z)) / sp.eta;

        if constexpr (!evalGradient)
            return s;
        else
        {
            const double ds = (z >= 0.0)
                ? 1.0 / (1.0 + std::exp(-z))
                : std::exp(z) / (1.0 + std::exp(z));
            return std::pair{s, ds};
        }
    }
};
