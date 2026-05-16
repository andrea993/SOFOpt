# SOFOpt

SOFOpt is a C++17 library for continuous-time structured static output-feedback
optimization.

Given a linear time-invariant plant

```text
dx/dt = A x + B u
y     = C x
u     = -Ky y
```

the library computes an output-feedback gain `Ky` and the equivalent full-state
gain `Kx = Ky * C`. The optimizer starts from the continuous-time LQR solution,
projects it onto the feasible output-feedback space, enforces optional sparsity
constraints on `Ky`, and minimizes an LQR-style quadratic cost while preserving
closed-loop stability through a soft spectral-abscissa constraint.

The primary public entry point is:

```cpp
SOF_Result SolveOutputFeedback(const SOF_Problem& problem);
```

## Features

- Static output-feedback synthesis for continuous-time LTI systems.
- Optional binary sparsity pattern for the output-feedback gain `Ky`.
- LQR-based initialization.
- SQP optimization through NLopt's `NLOPT_LD_SLSQP` backend.
- Eigen-based dense matrix API.
- Optional Riccati-solution preconditioning for improved numerical conditioning.
- C++ API, C-compatible shared-library ABI, and binary stdin/stdout executable
  interface.
- Debug executables with built-in examples.
- CPack ZIP packaging support.

## Repository Layout

```text
.
|-- CMakeLists.txt
|-- CMakePresets.json
|-- LICENSE.md
|-- README.md
|-- vcpkg.json
|-- Papers/
|   `-- 0609028v1.pdf
`-- src/
    |-- SOF_Problem.h              # Public problem/result structs
    |-- SOFOpt_Core.hpp/.cpp       # Main C++ API and input validation
    |-- SOFOpt_SQP.hpp/.cpp        # SQP optimizer implementation
    |-- ControlAlgebra.hpp/.cpp    # Riccati, Lyapunov, spectral utilities
    |-- SOFOpt_LibItf.h/.cpp       # C ABI wrapper
    |-- SOFOpt_ExecProtocol.hpp/.cpp
    |-- main.cpp                   # Binary protocol solver executable
    |-- debug_main.cpp             # Small built-in debug case
    `-- room_temperature_debug_main.cpp
```

## Dependencies

SOFOpt requires:

- CMake 3.21 or newer
- A C++17 compiler
- Eigen3
- LAPACK
- NLopt

The repository includes a `vcpkg.json` manifest with:

```json
{
  "dependencies": ["eigen3", "nlopt", "openblas", "lapack"]
}
```

On Linux, the libraries can also be supplied by the system package manager if
CMake can find `Eigen3`, `LAPACK`, and `NLopt`.

## Build

### Linux Preset

```sh
cmake --preset SOFOpt-Linux
cmake --build out/build/SOFOpt-Linux
```

The Linux preset uses `gcc`/`g++`, creates a Debug build, and writes artifacts to:

```text
out/build/SOFOpt-Linux/bin/
out/build/SOFOpt-Linux/lib/
```

### Windows MSVC Preset

```sh
cmake --preset SOFOpt-MSVC
cmake --build out/build/SOFOpt-MSVC --config Release
```

The MSVC preset uses Ninja, `cl`, `x64-windows`, and the vcpkg toolchain path
configured in `CMakePresets.json`. Adjust `CMAKE_TOOLCHAIN_FILE` if vcpkg is
installed elsewhere.

### Manual CMake Configuration

```sh
cmake -S . -B out/build/manual -DCMAKE_BUILD_TYPE=Release
cmake --build out/build/manual
```

When using vcpkg manually:

```sh
cmake -S . -B out/build/vcpkg \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build out/build/vcpkg
```

## Build Targets

CMake defines these targets:

| Target | Output | Description |
| --- | --- | --- |
| `SOFOpt_Lib` | `lib/SOFOpt_Lib` shared library | Shared library exposing the C ABI wrapper. |
| `SOFOpt_Exec` | `bin/solver_exec` | Command-line solver using the binary protocol. |
| `SOFOpt_Debug` | `bin/solver_debug` | Built-in small debug problem. |
| `SOFOpt_RoomDebug` | `bin/solver_room_debug` | Built-in room-temperature-style debug problem. |
| `SOFOpt_Objects` | object library | Internal object target used by the library and executables. |

## Quick Start: C++ API

Include the public API:

```cpp
#include "SOFOpt_Core.hpp"
```

Create a `SOF_Problem`, populate all required matrices, and call
`SolveOutputFeedback`:

```cpp
#include <Eigen/Dense>
#include <iostream>

#include "SOFOpt_Core.hpp"

int main()
{
    SOF_Problem problem;

    problem.A.resize(4, 4);
    problem.A << -1.0, 0.0, 1.0, 0.0,
                  0.0, -1.0, 0.0, -1.0,
                 -1.0, 0.0, -1.0, -1.0,
                  0.0, 1.0, -1.0, -1.0;

    problem.B.resize(4, 2);
    problem.B << 0.0, 0.0,
                 0.0, 0.0,
                 1.0, 0.0,
                 0.0, 1.0;

    problem.C.resize(2, 4);
    problem.C << 1.0, 1.0, 0.0, 0.0,
                 0.0, 1.0, 0.0, 0.0;

    problem.Q = Eigen::MatrixXd::Identity(4, 4);
    problem.R = Eigen::MatrixXd::Identity(2, 2);
    problem.X0.resize(4);
    problem.X0 << 1.0, 1.0, 1.0, 1.0;

    problem.Structure.resize(2, 2);
    problem.Structure << true, false,
                         false, true;

    problem.use_P_precond = true;
    problem.rho_alpha = 1e-6;
    problem.beta = 100.0;
    problem.r = 1e-5;
    problem.c = 500.0;

    const SOF_Result result = SolveOutputFeedback(problem);

    std::cout << "Kx:\n" << result.Kx << "\n\n";
    std::cout << "Ky:\n" << result.Ky << "\n\n";
    std::cout << "initial cost: " << result.init_cost << '\n';
    std::cout << "optimized cost: " << result.optim_cost << '\n';
    std::cout << "optimizer result: " << result.result << '\n';
}
```

The same example is available as `src/debug_main.cpp` and is built as
`solver_debug`.

## Problem Definition

`SOF_Problem` is declared in `src/SOF_Problem.h`:

```cpp
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
    double beta = 100.0;
    double r = 1e-5;
    double c = 500;
};
```

### Required Dimensions

Let:

- `Nx` be the number of states.
- `Nu` be the number of inputs.
- `Ny` be the number of measured outputs.

Then the matrices must satisfy:

| Field | Required size | Meaning |
| --- | --- | --- |
| `A` | `Nx x Nx` | Plant state matrix. |
| `B` | `Nx x Nu` | Plant input matrix. |
| `C` | `Ny x Nx` | Plant output matrix. |
| `Q` | `Nx x Nx` | State cost matrix. |
| `R` | `Nu x Nu` | Input cost matrix. |
| `X0` | `Nx` | Initial condition used in the cost evaluation. |
| `Structure` | `Nu x Ny` | Boolean sparsity mask for `Ky`. |

All fields are required. Empty matrices are rejected.

### Structure Mask

`Structure(i, j)` controls whether `Ky(i, j)` may be nonzero:

- `true`: this gain entry is free.
- `false`: this gain entry is constrained to zero.

For an unconstrained dense `Ky`, use a `Nu x Ny` matrix filled with `true`.

Example:

```cpp
problem.Structure =
    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>::Constant(Nu, Ny, true);
```

### Tuning Parameters

| Field | Default | Constraint | Purpose |
| --- | ---: | --- | --- |
| `use_P_precond` | `true` | boolean | Enables coordinate preconditioning using the Riccati solution. |
| `rho_alpha` | `1e-6` | finite, `>= 0` | Scales the soft spectral-abscissa stability bound. |
| `beta` | `100.0` | finite, `> 0` | Softmax sharpness for the soft spectral abscissa. Larger values approximate the max more tightly. |
| `r` | `1e-5` | finite, `>= 0` | Softplus cost parameter. |
| `c` | `500.0` | finite, `> 0` | Softplus cost parameter. |

The optimizer builds a closed-loop LQR reference, computes a soft
spectral-abscissa bound from that reference, and constrains the optimized
controller relative to that bound.

## Result Definition

`SOF_Result` is declared in `src/SOF_Problem.h`:

```cpp
struct SOF_Result
{
    Eigen::MatrixXd Kx;
    Eigen::MatrixXd Ky;
    double init_cost;
    double optim_cost;
    std::string result;
};
```

| Field | Meaning |
| --- | --- |
| `Kx` | Optimized full-state-equivalent gain, with size `Nu x Nx`. |
| `Ky` | Optimized static output-feedback gain, with size `Nu x Ny`. |
| `init_cost` | LQR reference cost in the optimization coordinates. |
| `optim_cost` | Cost obtained by the optimized `Kx` in the original coordinates. |
| `result` | NLopt result string, or an error message in wrapper contexts. |

The closed-loop matrix associated with the returned output-feedback controller is:

```cpp
Eigen::MatrixXd Acl = problem.A - problem.B * result.Ky * problem.C;
```

`result.Kx` is the optimized gain represented in state coordinates. In normal
use it satisfies the output-feedback constraints imposed by `C` and
`Structure`.

## Validation and Exceptions

`SolveOutputFeedback` validates:

- non-empty matrices and vector,
- dimension consistency,
- `Structure.rows() == B.cols()`,
- `Structure.cols() == C.rows()`,
- finite scalar tuning parameters,
- non-negative `rho_alpha` and `r`,
- positive `beta` and `c`.

Invalid inputs throw `std::invalid_argument`. Numerical failures from the
underlying Riccati, Lyapunov, Eigen, LAPACK, or NLopt routines may propagate as
exceptions or NLopt result strings depending on the call path.

## Running the Built-In Debug Programs

After building:

```sh
./out/build/SOFOpt-Linux/bin/solver_debug
./out/build/SOFOpt-Linux/bin/solver_room_debug
```

Both programs accept:

```text
--no-precond
--rho-alpha <value>
--help
```

Examples:

```sh
./out/build/SOFOpt-Linux/bin/solver_debug --rho-alpha 0.1
./out/build/SOFOpt-Linux/bin/solver_room_debug --no-precond
```

The debug executables print the LQR gain, optimized gains, costs, optimizer
result string, and closed-loop eigenvalue diagnostics.

## C ABI

The shared library exports:

```c
OptimOutput_C Optim_KxOut_C(
    const double* A, int A_rows, int A_cols,
    const double* B, int B_rows, int B_cols,
    const double* C, int C_rows, int C_cols,
    const double* Q, int Q_rows, int Q_cols,
    const double* R, int R_rows, int R_cols,
    const double* X0, int X0_size,
    const bool* Structure, int Structure_rows, int Structure_cols,
    bool use_P_precond,
    double rho_alpha,
    double beta,
    double r,
    double c);
```

The return type is:

```c
typedef struct {
    double* Kx;
    double* Ky;
    double init_cost;
    double optim_cost;
    char result[256];
} OptimOutput_C;
```

Important ABI details:

- Matrix buffers are interpreted in Eigen's default column-major layout.
- `Kx` has `B_cols * A_rows` doubles.
- `Ky` has `B_cols * C_rows` doubles.
- On success, `Kx` and `Ky` are allocated with `malloc`.
- The caller is responsible for freeing `Kx` and `Ky` with `free`.
- On failure, `Kx` and `Ky` are set to `NULL` and `result` contains the error
  message.

Minimal C-style cleanup:

```c
OptimOutput_C out = Optim_KxOut_C(/* arguments */);
if (out.Kx && out.Ky) {
    /* use out.Kx and out.Ky */
}
free(out.Kx);
free(out.Ky);
```

## Binary Executable Protocol

`solver_exec` reads one binary request from `stdin`, solves the problem, and
writes one binary response to `stdout`. It is useful for integrations that want
process isolation instead of linking to the shared library.

### Request Layout

All scalar values are written in the host's native endianness and ABI layout.
Matrix payloads are column-major doubles.

```text
uint32  magic = 0x3142464f
int32   A_rows
int32   A_cols
int32   B_rows
int32   B_cols
int32   C_rows
int32   C_cols
int32   Q_rows
int32   Q_cols
int32   R_rows
int32   R_cols
int32   X0_size
int32   Structure_rows
int32   Structure_cols
uint8   use_P_precond
double  rho_alpha
double  beta
double  r
double  c
double  A[A_rows * A_cols]
double  B[B_rows * B_cols]
double  C[C_rows * C_cols]
double  Q[Q_rows * Q_cols]
double  R[R_rows * R_cols]
double  X0[X0_size]
uint8   Structure[Structure_rows * Structure_cols]
```

`Structure` entries are interpreted as false when zero and true otherwise.

### Response Layout

```text
uint32  magic = 0x3142464f
int32   status              # 0 success, 1 failure
int32   Kx_rows             # 0 on failure
int32   Kx_cols             # 0 on failure
int32   Ky_rows             # 0 on failure
int32   Ky_cols             # 0 on failure
double  init_cost
double  optim_cost
int32   message_length      # max 4096
char    message[message_length]
double  Kx[Kx_rows * Kx_cols]   # success only
double  Ky[Ky_rows * Ky_cols]   # success only
```

The executable exits with:

- `0` on solver success,
- `1` on solver failure,
- `2` if the response cannot be written.

## Installation and Packaging

Install the library and executable:

```sh
cmake --install out/build/SOFOpt-Linux
```

The install rules place:

- executables in `bin`,
- shared libraries in `bin` on all platforms in the current CMake file,
- archives in `lib`.

Create a ZIP package with CPack:

```sh
cmake --build out/build/SOFOpt-Linux --target package
```

The package name follows:

```text
SOFOpt-<version>-<system>-<processor>.zip
```

## Numerical Notes

SOFOpt assumes a continuous-time plant and dense matrices. The optimization path
uses:

- continuous-time algebraic Riccati equation utilities for LQR initialization,
- Lyapunov equation solves for cost evaluation,
- Schur/eigenvalue-based spectral-abscissa utilities,
- a smooth softplus-modified cost around the stability boundary,
- NLopt SLSQP for constrained nonlinear optimization.

For best results:

- provide stabilizable and detectable systems,
- use symmetric positive semidefinite `Q`,
- use symmetric positive definite `R`,
- scale states, inputs, and outputs to comparable numerical ranges,
- keep `use_P_precond` enabled unless diagnosing conditioning behavior,
- start with a dense `Structure` mask before adding sparsity constraints.

## Limitations

- The public API currently targets dense `Eigen::MatrixXd` data.
- There is no installed CMake package config target yet; consumers should include
  the headers and link the built shared library or add this repository as a
  subdirectory.
- The binary protocol is intended for local integrations and does not define a
  portable cross-endian file format.
- The C ABI allocates output arrays but does not currently export a dedicated
  deallocation function; use the same C runtime `free` compatible with the
  library build.
- Automated tests are not currently defined in `CMakeLists.txt`.

## License

SOFOpt is distributed under the Mozilla Public License 2.0. See
`LICENSE.md` for the full license text.
