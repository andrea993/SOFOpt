cmake_minimum_required(VERSION 3.21)

project(SOFOpt
    VERSION 0.1.0
    LANGUAGES C CXX
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ------------------------------------------------------------------------------
# Build type
# ------------------------------------------------------------------------------

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# ------------------------------------------------------------------------------
# Platform checks
# ------------------------------------------------------------------------------

if(MINGW AND NOT CMAKE_GENERATOR MATCHES "MinGW Makefiles|Ninja")
    message(FATAL_ERROR
        "Please use a MinGW-compatible generator such as 'Ninja' or 'MinGW Makefiles'."
    )
endif()

# ------------------------------------------------------------------------------
# Sources
# ------------------------------------------------------------------------------

set(OF_SOURCES
    src/ControlAlgebra.cpp
    src/SOFOpt_SQP.cpp
    src/SOFOpt_LibItf.cpp
    src/SOFOpt_Core.cpp
    src/SOFOpt_ExecProtocol.cpp
)

set(OF_HEADERS
    src/ControlAlgebra.hpp
    src/SOFOpt_SQP.hpp
    src/SOF_Problem.h
    src/SOFOpt_LibItf.h
    src/SOFOpt_Core.hpp
    src/SOFOpt_ExecProtocol.hpp
)

# ------------------------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------------------------

find_package(Eigen3 REQUIRED)
find_package(LAPACK REQUIRED)
find_package(NLopt REQUIRED)


# ------------------------------------------------------------------------------
# Core compilation unit (OBJECT library)
# ------------------------------------------------------------------------------

add_library(SOFOpt_Objects OBJECT
    ${OF_SOURCES}
    ${OF_HEADERS}
)

set_target_properties(SOFOpt_Objects PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# Compile flags
target_compile_options(SOFOpt_Objects
    PRIVATE
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>>:-O2;-fno-math-errno;-fno-trapping-math>
)

# Link propagato
target_link_libraries(SOFOpt_Objects
    PUBLIC
        Eigen3::Eigen
        ${LAPACK_LIBRARIES}
        NLopt::nlopt
)

# ------------------------------------------------------------------------------
# Shared library
# ------------------------------------------------------------------------------

add_library(SOFOpt_Lib SHARED
    $<TARGET_OBJECTS:SOFOpt_Objects>
)

target_link_libraries(SOFOpt_Lib
    PRIVATE
        SOFOpt_Objects
)

set_target_properties(SOFOpt_Lib PROPERTIES
    OUTPUT_NAME SOFOpt_Lib
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ------------------------------------------------------------------------------
# Executables
# ------------------------------------------------------------------------------

add_executable(SOFOpt_Exec
    src/main.cpp
    $<TARGET_OBJECTS:SOFOpt_Objects>
)

target_link_libraries(SOFOpt_Exec
    PRIVATE
        SOFOpt_Objects
)

set_target_properties(SOFOpt_Exec PROPERTIES
    OUTPUT_NAME solver_exec
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

add_executable(SOFOpt_Debug
    src/debug_main.cpp
    $<TARGET_OBJECTS:SOFOpt_Objects>
)

target_link_libraries(SOFOpt_Debug
    PRIVATE
        SOFOpt_Objects
)

set_target_properties(SOFOpt_Debug PROPERTIES
    OUTPUT_NAME solver_debug
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

add_executable(SOFOpt_RoomDebug
    src/room_temperature_debug_main.cpp
    $<TARGET_OBJECTS:SOFOpt_Objects>
)

target_link_libraries(SOFOpt_RoomDebug
    PRIVATE
        SOFOpt_Objects
)

set_target_properties(SOFOpt_RoomDebug PROPERTIES
    OUTPUT_NAME solver_room_debug
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ------------------------------------------------------------------------------
# IDE organization
# ------------------------------------------------------------------------------

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES
    ${OF_SOURCES}
    ${OF_HEADERS}
    src/main.cpp
    src/debug_main.cpp
    src/room_temperature_debug_main.cpp
)

# ------------------------------------------------------------------------------
# Portable package
# ------------------------------------------------------------------------------

include(GNUInstallDirs)

install(TARGETS SOFOpt_Exec SOFOpt_Lib
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION bin
    ARCHIVE DESTINATION lib
)


if(WIN32)
    if(MSVC)
        set(CMAKE_INSTALL_SYSTEM_RUNTIME_DESTINATION bin)
        include(InstallRequiredSystemLibraries)
    endif()

    install(FILES
        $<TARGET_RUNTIME_DLLS:SOFOpt_Exec>
        $<TARGET_RUNTIME_DLLS:SOFOpt_Lib>
        DESTINATION bin
        OPTIONAL
    )
endif()

set(CPACK_GENERATOR "ZIP")
set(CPACK_PACKAGE_NAME "SOFOpt")
set(CPACK_PACKAGE_VENDOR "SOFOpt")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${PROJECT_VERSION}-${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
include(CPack)
