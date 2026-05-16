#pragma once

#include <cstdint>
#include <iosfwd>
#include <string>

#include "SOFOpt_Core.hpp"

namespace SOFOpt_ExecProtocol
{
constexpr std::uint32_t kMagic = 0x3142464fU;
constexpr std::int32_t kSuccess = 0;
constexpr std::int32_t kFailure = 1;

struct SolverResponse
{
    std::int32_t status = kFailure;
    SOF_Result result;
    std::string message;
};

SOF_Problem ReadProblem(std::istream& input);
void WriteResponse(std::ostream& output, const SolverResponse& response);
}
