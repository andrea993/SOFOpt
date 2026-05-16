#include "ControlAlgebra.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

extern "C" {
void dgees_(const char* jobvs,
            const char* sort,
            int (*select)(double*, double*),
            const int* n,
            double* a,
            const int* lda,
            int* sdim,
            double* wr,
            double* wi,
            double* vs,
            const int* ldvs,
            double* work,
            const int* lwork,
            int* bwork,
            int* info);

void dtrsyl_(const char* trana,
             const char* tranb,
             const int* isgn,
             const int* m,
             const int* n,
             const double* a,
             const int* lda,
             const double* b,
             const int* ldb,
             double* c,
             const int* ldc,
             double* scale,
             int* info);

void dtrevc_(const char* side,
             const char* howmny,
             int* select,
             const int* n,
             const double* t,
             const int* ldt,
             double* vl,
             const int* ldvl,
             double* vr,
             const int* ldvr,
             const int* mm,
             int* m,
             double* work,
             int* info);
}

namespace
{
struct RealSchurDecomposition
{
    MatrixXd T;
    MatrixXd U;
    VectorXd wr;
    VectorXd wi;
};

int SelectStableEigenvalue(double* wr, double*)
{
    return *wr < 0.0;
}

RealSchurDecomposition LapackRealSchur(const MatrixXd& A, const bool sort_stable, int* selected_count = nullptr)
{
    const int n = static_cast<int>(A.rows());
    if (A.cols() != n) {
        throw std::invalid_argument("LapackRealSchur requires a square matrix");
    }
    if (!A.allFinite()) {
        throw std::runtime_error("LapackRealSchur received a matrix with non-finite entries");
    }

    MatrixXd schur_input = A;
    MatrixXd schur_vectors(n, n);
    VectorXd wr(n);
    VectorXd wi(n);
    int sdim = 0;
    int info = 0;
    int lwork = -1;
    double work_query = 0.0;
    std::vector<int> bwork(std::max(1, n));

    const char jobvs = 'V';
    const char sort = sort_stable ? 'S' : 'N';
    auto select = sort_stable ? SelectStableEigenvalue : nullptr;

    dgees_(&jobvs,
           &sort,
           select,
           &n,
           schur_input.data(),
           &n,
           &sdim,
           wr.data(),
           wi.data(),
           schur_vectors.data(),
           &n,
           &work_query,
           &lwork,
           bwork.data(),
           &info);
    if (info != 0) {
        throw std::runtime_error("LAPACK dgees workspace query failed, info=" + std::to_string(info));
    }

    lwork = std::max(1, static_cast<int>(work_query));
    std::vector<double> work(static_cast<std::size_t>(lwork));

    dgees_(&jobvs,
           &sort,
           select,
           &n,
           schur_input.data(),
           &n,
           &sdim,
           wr.data(),
           wi.data(),
           schur_vectors.data(),
           &n,
           work.data(),
           &lwork,
           bwork.data(),
           &info);
    if (info != 0) {
        throw std::runtime_error(
            "LAPACK dgees failed to compute Schur decomposition, sort="
            + std::string(1, sort)
            + ", info="
            + std::to_string(info)
            + ", max_abs="
            + std::to_string(A.cwiseAbs().maxCoeff())
            + ", fro_norm="
            + std::to_string(A.norm()));
    }

    if (selected_count != nullptr) {
        *selected_count = sdim;
    }

    return RealSchurDecomposition{schur_input, schur_vectors, wr, wi};
}

MatrixXd RealSchurForm(const MatrixXd& A, MatrixXd& schur_vectors)
{
    auto schur = LapackRealSchur(A, false);
    schur_vectors = std::move(schur.U);
    return schur.T;
}

std::pair<MatrixXcd, MatrixXcd> SchurLeftRightEigenvectors(const RealSchurDecomposition& schur)
{
    const int n = static_cast<int>(schur.T.rows());
    MatrixXd vl(n, n);
    MatrixXd vr(n, n);
    std::vector<int> select(std::max(1, n), 0);
    std::vector<double> work(static_cast<std::size_t>(3 * std::max(1, n)));
    int m = 0;
    int info = 0;
    const int mm = n;
    const char side = 'B';
    const char howmny = 'A';

    dtrevc_(&side,
            &howmny,
            select.data(),
            &n,
            schur.T.data(),
            &n,
            vl.data(),
            &n,
            vr.data(),
            &n,
            &mm,
            &m,
            work.data(),
            &info);
    if (info != 0) {
        throw std::runtime_error("LAPACK dtrevc failed to compute Schur eigenvectors, info=" + std::to_string(info));
    }

    MatrixXcd left(n, n);
    MatrixXcd right(n, n);
    const MatrixXd& U = schur.U;
    for (int i = 0; i < n; ++i) {
        if (schur.wi(i) == 0.0) {
            left.col(i) = (U * vl.col(i)).cast<std::complex<double>>();
            right.col(i) = (U * vr.col(i)).cast<std::complex<double>>();
            continue;
        }

        if (schur.wi(i) > 0.0) {
            left.col(i) = (U * vl.col(i)).cast<std::complex<double>>()
                        + std::complex<double>(0.0, 1.0) * (U * vl.col(i + 1)).cast<std::complex<double>>();
            right.col(i) = (U * vr.col(i)).cast<std::complex<double>>()
                         + std::complex<double>(0.0, 1.0) * (U * vr.col(i + 1)).cast<std::complex<double>>();
        } else {
            left.col(i) = left.col(i - 1).conjugate();
            right.col(i) = right.col(i - 1).conjugate();
        }
    }

    return {left, right};
}

MatrixXd EigenvalueRealPartGradient(const VectorXcd& left, const VectorXcd& right)
{
    const std::complex<double> denom = left.dot(right);
    if (std::abs(denom) == 0.0) {
        throw std::runtime_error("Cannot normalize Schur eigenvectors for spectral gradient");
    }

    return (left.conjugate() * right.transpose() / denom).real();
}

} // namespace

MatrixXd ControlAlgebra::Slyap(const MatrixXd &A, const MatrixXd &Q)
{
    const int n = static_cast<int>(A.rows());
    if (A.cols() != n || Q.rows() != n || Q.cols() != n) {
        throw std::invalid_argument("Slyap requires square A and matching square Q");
    }

    MatrixXd U;
    MatrixXd T;
    try {
        T = RealSchurForm(A, U);
    } catch (const std::exception& error) {
        throw std::runtime_error(std::string("Slyap Schur failed: ") + error.what());
    }
    MatrixXd Y = -U.transpose() * Q * U;

    const char trana = 'N';
    const char tranb = 'T';
    const int isgn = 1;
    double scale = 1.0;
    int info = 0;

    dtrsyl_(&trana,
            &tranb,
            &isgn,
            &n,
            &n,
            T.data(),
            &n,
            T.data(),
            &n,
            Y.data(),
            &n,
            &scale,
            &info);
    if (info < 0) {
        throw std::runtime_error("LAPACK dtrsyl received an invalid argument");
    }
    if (info > 1) {
        throw std::runtime_error("LAPACK dtrsyl failed to solve Lyapunov equation");
    }

    return U * (Y / scale) * U.transpose();
}

MatrixXd ControlAlgebra::SlyapEigen(const MatrixXd &A, const MatrixXd &Q)
{
    const auto N = A.rows();
    EigenSolver<MatrixXd> es;
    es.compute(A, true); // Ensure eigenvectors are computed

    if (!es.info() == Success)
    {
        throw std::runtime_error("EigenSolver failed to compute eigenvalues and eigenvectors.");
    }

    const auto lambda = es.eigenvalues();
    const auto U = es.eigenvectors();

    const auto Uinv = U.inverse();
    const auto W = Uinv * Q * (Uinv.transpose());

    auto Z = MatrixXcd::Zero(N, N).eval();
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            Z(i, j) = W(i, j) / (lambda(i) + lambda(j));
        }
    }

    return (-U * Z * U.transpose()).real();
}

double ControlAlgebra::SpectralAbscissaSchurValue(const MatrixXd &A)
{
    RealSchurDecomposition schur;
    try {
        schur = LapackRealSchur(A, false);
    } catch (const std::exception& error) {
        throw std::runtime_error(std::string("SpectralAbscissaSchurValue failed: ") + error.what());
    }
    return schur.wr.maxCoeff();
}

std::pair<double, MatrixXd> ControlAlgebra::SpectralAbscissaSchurGradient(const MatrixXd &A)
{
    RealSchurDecomposition schur;
    try {
        schur = LapackRealSchur(A, false);
    } catch (const std::exception& error) {
        throw std::runtime_error(std::string("SpectralAbscissaSchurGradient failed: ") + error.what());
    }
    Eigen::Index max_idx = 0;
    const double alpha = schur.wr.maxCoeff(&max_idx);
    const auto [left, right] = SchurLeftRightEigenvectors(schur);
    return {alpha, EigenvalueRealPartGradient(left.col(max_idx), right.col(max_idx))};
}

double ControlAlgebra::SoftSpectralAbscissaSchurValue(const MatrixXd& A, const double beta)
{
    /*
        alpha_beta(A) = (1 / beta) log sum_i exp(beta Re(lambda_i(A)))

        Smooth approximation of:

            alpha(A) = max_i Re(lambda_i(A))

        Bound:

            alpha(A) <= alpha_beta(A) <= alpha(A) + log(n) / beta

        Note: smooths the max, but not eigenvalue multiplicities.
    */

    if (beta <= 0.0)
        throw std::invalid_argument("beta must be positive.");

    RealSchurDecomposition schur;

    try
    {
        // Real Schur decomposition. schur.wr contains Re(lambda_i(A)).
        schur = LapackRealSchur(A, false);
    }
    catch (const std::exception& error)
    {
        throw std::runtime_error(
            std::string("SoftSpectralAbscissaSchurValue failed: ") +
            error.what()
        );
    }

    /*
        Stable log-sum-exp:

            r_max = max_i r_i

            alpha_beta =
                r_max + (1 / beta) log sum_i exp(beta (r_i - r_max))

        where r_i = Re(lambda_i(A)).
    */

    const double r_max = schur.wr.maxCoeff();

    const VectorXd exp_vals =
        (beta * (schur.wr.array() - r_max)).exp();

    const double sum_exp = exp_vals.sum();

    return r_max + std::log(sum_exp) / beta;
}

std::pair<double, MatrixXd> ControlAlgebra::SoftSpectralAbscissaSchurGradient(const MatrixXd &A, const double beta)
{
    RealSchurDecomposition schur;
    try {
        schur = LapackRealSchur(A, false);
    } catch (const std::exception& error) {
        throw std::runtime_error(std::string("SoftSpectralAbscissaSchurGradient failed: ") + error.what());
    }
    const double max_val = schur.wr.maxCoeff();
    const VectorXd exp_vals = (beta * (schur.wr.array() - max_val)).exp();
    const double sum_exp = exp_vals.sum();
    const VectorXd weights = exp_vals / sum_exp;
    const double alpha = max_val + std::log(sum_exp) / beta;

    const auto [left, right] = SchurLeftRightEigenvectors(schur);
    MatrixXd gradient = MatrixXd::Zero(A.rows(), A.cols());
    for (int i = 0; i < A.rows(); ++i) {
        gradient += weights(i) * EigenvalueRealPartGradient(left.col(i), right.col(i));
    }

    return {alpha, gradient};
}

MatrixXd ControlAlgebra::AREc(const MatrixXd &A, const MatrixXd &B, const MatrixXd &C)
{
    const int n = static_cast<int>(A.rows());
    if (A.cols() != n || B.rows() != n || B.cols() != n || C.rows() != n || C.cols() != n) {
        throw std::invalid_argument("AREc requires square A, B, C with matching dimensions");
    }

    MatrixXd H(A.rows() + C.rows(), A.cols() + B.cols());
    H << A, B,
         C, -A.transpose();

    int selected_count = 0;
    MatrixXd U;
    try {
        U = LapackRealSchur(H, true, &selected_count).U;
    } catch (const std::exception& error) {
        throw std::runtime_error(std::string("AREc Schur failed: ") + error.what());
    }
    if (selected_count != n) {
        throw std::runtime_error("AREc Schur decomposition did not isolate the stable invariant subspace");
    }

    auto U_11 = U(seq(0, n - 1), seq(0, n - 1)).eval();
    auto U_21 = U(seq(n, 2 * n - 1), seq(0, n - 1)).eval();

    MatrixXd S = U_11.transpose().colPivHouseholderQr().solve(U_21.transpose()).transpose();
    return 0.5 * (S + S.transpose());
}


MatrixXd ControlAlgebra::AREcEigen(const MatrixXd &A, const MatrixXd &B, const MatrixXd &C)
{
    MatrixXd H(A.rows() + C.rows(), A.cols() + B.cols());
    H << A, B,
         C, -A.transpose();

    auto n = H.rows()/2;

    auto SE = ControlAlgebra::SortEig(H);
    auto lambda = SE.first;
    auto V = SE.second;

    auto X_IN = V(seq(0, n - 1), seq(0, n - 1));
    auto Y_IN = V(seq(n, 2*n - 1), seq(0, n - 1));
    
    return (Y_IN * X_IN.inverse()).real();
}

std::pair<MatrixXd, MatrixXd> ControlAlgebra::K_LQRc(const MatrixXd &A, const MatrixXd &B, const MatrixXd &Qx, const MatrixXd &Qu)
{
    auto Qu_inv = Qu.inverse();
    auto B_h = -B * Qu_inv * B.transpose();
    auto C_h = -Qx;

    auto S = AREc(A, B_h, C_h);
    auto K = Qu_inv * B.transpose() * S;

    return std::make_pair(K, S);
}

std::pair<MatrixXd, MatrixXd> ControlAlgebra::K_LQRc(const MatrixXd &A, const MatrixXd &B, const MatrixXd &Qx, const MatrixXd &Qu, const MatrixXd &Qxu)
{
    auto Qu_inv = Qu.inverse();

    auto A_h = A - B * Qu_inv * Qxu.transpose();
    auto B_h = -B * Qu_inv * B.transpose();
    auto C_h = -(Qx - Qxu * Qu_inv * Qxu.transpose());

    auto S = AREc(A_h, B_h, C_h);
    auto K = Qu_inv * (B.transpose() * S + Qxu.transpose());

    return std::make_pair(K, S);
}


std::pair<MatrixXcd, MatrixXcd> ControlAlgebra::SortEig(const MatrixXd &A) {
    EigenSolver<MatrixXd> es(A);
    const auto eigenvals = es.eigenvalues();
    const auto eigenvecs = es.eigenvectors();
    
    const int n = eigenvals.size();
    std::vector<std::pair<std::complex<double>, int>> eigenval_idx;
    for (int i = 0; i < n; ++i) {
        eigenval_idx.push_back({eigenvals(i), i});
    }
    
    std::sort(eigenval_idx.begin(), eigenval_idx.end(), 
        [](const auto& a, const auto& b) { return a.first.real() < b.first.real(); });
    
    MatrixXcd sorted_vals = MatrixXcd::Zero(n, 1);
    MatrixXcd sorted_vecs = MatrixXcd::Zero(n, n);
    
    for (int i = 0; i < n; ++i) {
        sorted_vals(i) = eigenval_idx[i].first;
        sorted_vecs.col(i) = eigenvecs.col(eigenval_idx[i].second);
    }
    
    return {sorted_vals, sorted_vecs};
}
