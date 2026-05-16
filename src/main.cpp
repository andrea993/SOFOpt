#include <exception>
#include <iostream>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

#include "SOFOpt_Core.hpp"
#include "SOFOpt_ExecProtocol.hpp"

int main()
{
#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif

    SOFOpt_ExecProtocol::SolverResponse response;

    try {
        const auto problem = SOFOpt_ExecProtocol::ReadProblem(std::cin);
        response.status = SOFOpt_ExecProtocol::kSuccess;
        response.result = SolveOutputFeedback(problem);
        response.message = response.result.result;
    } catch (const std::exception& error) {
        response.status = SOFOpt_ExecProtocol::kFailure;
        response.message = error.what();
    } catch (...) {
        response.status = SOFOpt_ExecProtocol::kFailure;
        response.message = "Unknown non-standard exception";
    }

    try {
        SOFOpt_ExecProtocol::WriteResponse(std::cout, response);
        std::cout.flush();
    } catch (...) {
        return 2;
    }

    return response.status == SOFOpt_ExecProtocol::kSuccess ? 0 : 1;
}
