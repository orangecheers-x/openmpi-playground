cmake_minimum_required(VERSION 3.26)
project(openmpi)

set(CMAKE_CXX_STANDARD 23)

find_package(MPI REQUIRED)
message(STATUS "Run: ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} EXECUTABLE ${MPIEXEC_POSTFLAGS} ARGS")

add_executable(openmpi mybroadcast.cpp)

target_link_libraries(openmpi PRIVATE MPI::MPI_CXX)