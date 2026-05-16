#ifndef SOLVER_FROM_LQR_C_H
#define SOLVER_FROM_LQR_C_H

#ifdef _WIN32
  #define DLL_EXPORT __declspec(dllexport)
#else
  #define DLL_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double* Kx;
    double* Ky;
    double init_cost;
    double optim_cost;
    char result[256];
} OptimOutput_C;

DLL_EXPORT OptimOutput_C Optim_KxOut_C(
    const double* A, int A_rows, int A_cols,
    const double* B, int B_rows, int B_cols,
    const double* C, int C_rows, int C_cols,
    const double* Q, int Q_rows, int Q_cols,
    const double* R, int R_rows, int R_cols,
    const double* X0, int X0_size,
    const bool* Structure, int Structure_rows, int Structure_cols,
    bool use_P_precond,
    double rho_alpha,
    double beta, // Softmax spectral absicssa
    double r, //Softplus cost
    double c //Softplus cost
  );

#ifdef __cplusplus
}
#endif

#endif // SOLVER_FROM_LQR_C_H
